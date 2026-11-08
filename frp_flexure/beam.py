"""Partial-interaction beam equilibrium and nonlinear section models."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares, minimize
from threadpoolctl import threadpool_limits
from frp_flexure.data import ROOT, load_strains
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
threadpool_limits(limits=1, user_api="blas")
H, B, TF, TW = 203.75, 103.14, 8.48, 5.78
AS = 2 * B * TF + (H - 2 * TF) * TW
IS = (B * H**3 - (B - TW) * (H - 2 * TF)**3) / 12
ES, EF = 180., 62.19
EI, MY = ES * IS, .465 * IS / (H / 2)
FY = .465
TRAIN_LOADS = np.arange(50., 351., 10.)
CALIBRATION_LOADS = TRAIN_LOADS[TRAIN_LOADS <= 200]
EVALUATION_LOADS = TRAIN_LOADS[TRAIN_LOADS >= 250]
HELDOUT = ("U65S100", "U65D35", "ST65S100")
WIDTH = 4
BOUNDS = np.log([[1e-5, 1e4]] * WIDTH + [[1., 80.]] * WIDTH
                + [[.25, 2.]] * WIDTH + [[.1, 10.]] * 2)
LAW = json.loads((ROOT / "data/beams/connector_law.json").read_text(encoding="utf-8"))

def designs():
    with (ROOT / "data/beams/configurations.csv").open(encoding="utf-8", newline="") as stream:
        return {row["specimen"]: row for row in csv.DictReader(stream)}


def geometry(row, subdivisions=16):
    """Half beam, with an exact element boundary at the applied point load."""
    half_patch = float(row['patch_length_mm']) / 2
    thick = float(row['patch_thickness_mm'])
    nodes = np.r_[np.linspace(0, 300, subdivisions + 1),
                  np.linspace(300, half_patch, subdivisions + 1)[1:]]
    ne = len(nodes) - 1
    lengths = np.diff(nodes)
    gp = np.array([-.5773502691896257, .5773502691896257])
    shape = np.column_stack(((1-gp)/2, (1+gp)/2))
    x = (nodes[:-1, None] + lengths[:, None] * (gp[None, :]+1)/2).ravel()
    weights = np.repeat(lengths/2, 2)
    interpolation = np.zeros((2*ne, ne+1))
    derivative = np.zeros((ne, ne+1))
    for e in range(ne):
        interpolation[2*e:2*e+2, e:e+2] = shape
        derivative[e, e:e+2] = [-1/lengths[e], 1/lengths[e]]
    g, w = np.polynomial.legendre.leggauss(24)
    outside = half_patch + (900-half_patch)*(g+1)/2
    return {
        'x': x, 'weight': weights, 'shape': interpolation[:, :-1],
        'gradient': derivative[:, :-1], 'length': lengths,
        'outside_x': outside, 'outside_weight': w*(900-half_patch)/2,
        'eccentricity': H/2+thick/2,
        'axial_compliance': 1/(ES*AS)+1/(EF*101.6*thick),
        'bolt_density': float(row['bolt_count']) / (2*half_patch),
        'thickness_ratio': thick/3.175,
    }


class BeamBatch:
    def __init__(self, names, loads, subdivisions=16):
        table = designs()
        self.names = list(names)
        self.loads = np.asarray(loads, dtype=float)
        geo = [geometry(table[n], subdivisions) for n in names]
        self.g = {k: torch.tensor(np.stack([v[k] for v in geo]))
                  .repeat_interleave(len(loads), dim=0) for k in geo[0]}
        self.p = torch.tensor(np.tile(loads, len(names)))
        self.moment = self.p[:, None] * torch.minimum(
            torch.full_like(self.g['x'], 300.), (900-self.g['x'])/2)
        self.outside_moment = self.p[:, None] * (900-self.g['outside_x'])/2
        self.elastic_scale = None

    def steel(self, moment, par, x=None):
        c, a, exponent = par[:3]
        if self.elastic_scale is not None:
            c = self.elastic_scale[:, None]
            a = a*c
        ratio = torch.abs(moment) / MY
        power = ratio.pow(exponent-1)
        curvature = (c + a*power) * moment/EI
        tangent = (c + a*exponent*power) / EI
        energy = c*moment.square()/(2*EI) + a*MY**2/EI*ratio.pow(exponent+1)/(exponent+1)
        return curvature, tangent, energy

    def system(self, force, par):
        g = self.g
        n = torch.einsum('bqi,bi->bq', g['shape'], force)
        q = torch.einsum('bei,bi->be', g['gradient'], force)
        steel_moment = self.moment - g['eccentricity'][:, None] * n
        curvature, tangent, energy = self.steel(steel_moment, par, g['x'])
        h = g['axial_compliance'][:, None]
        e = g['eccentricity'][:, None]
        slip, compliance, connector_energy = self.connector(q, par)
        local_gradient = -e*curvature+h*n
        local_hessian = e.square()*tangent+h
        grad = torch.einsum('bqi,bq->bi', g['shape'], g['weight']*local_gradient)
        grad = grad + torch.einsum('bei,be->bi', g['gradient'], g['length']*slip)
        hess = torch.einsum('bqi,bq,bqj->bij', g['shape'], g['weight']*local_hessian, g['shape'])
        hess = hess + torch.einsum('bei,be,bej->bij', g['gradient'], g['length']*compliance, g['gradient'])
        value = torch.sum(g['weight']*(energy+.5*h*n.square()), dim=1)
        value = value + torch.sum(g['length']*connector_energy, dim=1)
        return grad, hess, value, curvature

    def equilibrium(self, par):
        with torch.no_grad():
            force = torch.zeros((len(self.p), self.g['shape'].shape[-1]))
            for _ in range(100):
                grad, hess, value, _ = self.system(force, par)
                step = torch.linalg.solve(hess, grad[..., None]).squeeze(-1)
                if step.abs().max().item() < 1e-8:
                    break
                alpha = torch.ones_like(value)
                for _ in range(24):
                    proposed = force-alpha[:, None]*step
                    _, _, trial, _ = self.system(proposed, par)
                    bad = trial > value-1e-4*alpha*torch.sum(grad*step, dim=1)+1e-11
                    if not bad.any():
                        break
                    alpha[bad] *= .5
                force -= alpha[:, None]*step
            else:
                raise RuntimeError('Partial-interaction equilibrium did not converge')
        if par.requires_grad:
            grad, hess, _, _ = self.system(force, par)
            force = force-torch.linalg.solve(hess.detach(), grad[..., None]).squeeze(-1)
        return force

    def predict(self, log_parameters):
        par = torch.exp(log_parameters)
        force = self.equilibrium(par)
        _, _, _, curvature = self.system(force, par)
        outside_curvature = self.steel(self.outside_moment, par, self.g['outside_x'])[0]
        inside = torch.sum(self.g['weight']*(900-self.g['x'])*curvature, dim=1)
        outside = torch.sum(self.g['outside_weight']*(900-self.g['outside_x'])*outside_curvature, dim=1)
        return (inside+outside).reshape(len(self.names), len(self.loads))


def observed_compliances(inputs):
    p = CALIBRATION_LOADS
    return {n: float(p@np.asarray(v['equal_force_mean'])[:len(p)]/(p@p))
            for n, v in inputs.items()}


class ConditionedBatch:
    def predict(self, theta):
        full = torch.cat((torch.zeros(1), theta))
        linear = torch.cat((torch.zeros(1), torch.full((1,), -60.), theta[1:]))
        c = self.observed*EI/103500000.
        for _ in range(9):
            self.elastic.elastic_scale = c
            value = self.elastic.predict(linear)[:, 0]
            c = c*self.observed/value
        self.response.elastic_scale = c.repeat_interleave(self.repeat)
        return self.response.predict(full)


def reinforcement_strain(batch, theta, distances=(100., 400.)):
    """Evaluate single-strip underside strains at the requested gauge positions."""
    batch.predict(theta)
    full = torch.exp(torch.cat((torch.zeros(1), theta)))
    b = batch.response
    force = b.equilibrium(full)
    nodes = torch.cat((torch.zeros((1, 1)), b.g['length'].cumsum(1)), dim=1)[0]
    complete_force = torch.cat((force[0], torch.zeros(1)))
    values = []
    for x in distances:
        left = int(torch.searchsorted(nodes, torch.tensor(x)).item())-1
        weight = (x-nodes[left])/(nodes[left+1]-nodes[left])
        n = complete_force[left]*(1-weight)+complete_force[left+1]*weight
        m = b.p[0]*min(300., (900-x)/2)-b.g['eccentricity'][0]*n
        curvature = b.steel(m.reshape(1, 1), full, torch.tensor([[x]]))[0][0, 0]
        values.append(1e6*(n/(EF*101.6*3.175*b.g['thickness_ratio'][0])+3.175*b.g['thickness_ratio'][0]/2*curvature))
    return torch.stack(values)


class BearingBatch(BeamBatch):
    def connector(self, q, par):
        rho = self.g['bolt_density'][:, None]
        thick = self.g['thickness_ratio'][:, None]
        demand = torch.abs(q/rho)
        reference = torch.tensor(LAW['force_kn'])[None, :]
        # par[3] and par[4] scale joint slip and force after the source prior.
        force_knots = torch.where(reference <= 1.25, reference,
                                  1.25+(reference-1.25)*thick)*par[4]
        slip_knots = torch.tensor(LAW['slip_mm'])*par[3]
        slip = torch.zeros_like(q)
        energy = torch.zeros_like(q)
        compliance = torch.zeros_like(q)
        for j in range(4):
            lo, hi = force_knots[:, j:j+1], force_knots[:, j+1:j+2]
            length = hi-lo
            covered = torch.clamp_min(demand-lo, 0.)
            if j < 3:
                covered = torch.minimum(covered, length)
            slope = (slip_knots[j+1]-slip_knots[j])/length
            energy += slip_knots[j]*covered+.5*slope*covered.square()
            active = (demand >= lo) & ((demand < hi) if j < 3 else torch.ones_like(q, dtype=torch.bool))
            slip += torch.where(active, slip_knots[j]+slope*covered, 0.)
            compliance += torch.where(active, slope/rho, 0.)
        return torch.sign(q)*slip, compliance, rho*energy


class ConditionedBearing(ConditionedBatch):
    def __init__(self, names, loads, compliances):
        self.elastic = BearingBatch(names, [1.])
        self.response = BearingBatch(names, loads)
        self.observed = torch.tensor([compliances[n] for n in names])
        self.repeat = len(loads)


def fit_moment(names, inputs, compliance, joint):
    batch = ConditionedBearing(names, TRAIN_LOADS, compliance)
    gauge = ConditionedBearing(['U90S100'], [310.], compliance)
    target = torch.tensor([inputs[n]['equal_force_mean'] for n in names])
    def objective(theta):
        p = torch.tensor(theta, requires_grad=True)
        loss = ((batch.predict(p)-target)/10).square().mean()
        if joint:
            loss = loss+((reinforcement_strain(gauge, p)-torch.tensor(load_strains('U90S100')['microstrain']))/2000).square().mean()
        loss.backward()
        return loss.item(), p.grad.numpy()
    start = np.log([.2, 14., 1., 1.])
    bounds = np.log([[1e-4, 10.], [2., 40.], [.1, 10.], [.1, 10.]])
    solved = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 30})
    return {'parameters': dict(zip(('inelastic_amplitude','moment_exponent',
                                    'joint_slip_scale','joint_force_scale'), np.exp(solved.x).tolist())),
            'log_parameters': solved.x.tolist(), 'objective': float(solved.fun),
            'training': names, 'success': bool(solved.success), 'message': str(solved.message),
            'training_strain_microstrain': reinforcement_strain(gauge, torch.tensor(solved.x)).detach().tolist()}


def section_quadrature(order):
    xi, wi = np.polynomial.legendre.leggauss(order)
    y, area = [], []
    for lo, hi, width in ((-H/2, -H/2+TF, B), (-H/2+TF, H/2-TF, TW), (H/2-TF, H/2, B)):
        y.extend(lo+(xi+1)*(hi-lo)/2)
        area.extend(wi*(hi-lo)/2*width)
    return torch.tensor(y), torch.tensor(area)


class CoupledBatch(BearingBatch):
    def __init__(self, names, loads, subdivisions=16, section_order=4):
        super().__init__(names, loads, subdivisions)
        self.fiber_y, self.fiber_area = section_quadrature(section_order)

    def material(self, stress, par):
        c = self.elastic_scale[:, None, None] if self.elastic_scale is not None else par[0]
        a, power = par[1:3]
        ratio = torch.abs(stress)/FY
        strain = c/ES*stress*(1+a*ratio.pow(power-1))
        tangent = c/ES*(1+a*power*ratio.pow(power-1))
        energy = c/ES*(stress.square()/2+a*FY**2*ratio.pow(power+1)/(power+1))
        return strain, tangent, energy

    def steel(self, moment, par, x=None):
        y, area = self.fiber_y, self.fiber_area
        stress = moment[..., None]*y/IS
        strain, tangent, energy = self.material(stress, par)
        curvature = torch.sum(area*y/IS*strain, dim=-1)
        derivative = torch.sum(area*(y/IS).square()*tangent, dim=-1)
        return curvature, derivative, torch.sum(area*energy, dim=-1)

    def section_response(self, n, moment, e, par):
        y, area = self.fiber_y, self.fiber_area
        stress = -n[..., None]/AS+moment[..., None]*y/IS
        strain, tangent, density = self.material(stress, par)
        mode = -1/AS-e[..., None]*y/IS
        gradient = torch.sum(area*mode*strain, dim=-1)
        derivative = torch.sum(area*mode.square()*tangent, dim=-1)
        curvature = torch.sum(area*y/IS*strain, dim=-1)
        return torch.sum(area*density, dim=-1), curvature, gradient, derivative

    def system(self, force, par):
        g = self.g
        n = torch.einsum('bqi,bi->bq', g['shape'], force)
        q = torch.einsum('bei,bi->be', g['gradient'], force)
        e = g['eccentricity'][:, None]
        energy, curvature, local_gradient, local_tangent = self.section_response(
            n, self.moment-e*n, e, par)
        hfrp_compliance = 1/(EF*101.6*3.175*g['thickness_ratio'][:, None])
        local_gradient = local_gradient+hfrp_compliance*n
        local_tangent = local_tangent+hfrp_compliance
        energy = energy+hfrp_compliance*n.square()/2
        slip, compliance, connector_energy = self.connector(q, par)
        grad = torch.einsum('bqi,bq->bi', g['shape'], g['weight']*local_gradient)
        grad += torch.einsum('bei,be->bi', g['gradient'], g['length']*slip)
        hess = torch.einsum('bqi,bq,bqj->bij', g['shape'], g['weight']*local_tangent, g['shape'])
        hess += torch.einsum('bei,be,bej->bij', g['gradient'], g['length']*compliance, g['gradient'])
        value = torch.sum(g['weight']*energy, dim=1)+torch.sum(g['length']*connector_energy, dim=1)
        return grad, hess, value, curvature


class SeparateBatch(CoupledBatch):
    """Retain the pure axial and pure bending laws, omit their interaction."""
    def section_response(self, n, moment, e, par):
        axial_strain, axial_tangent, axial_energy = self.material(-n[..., None]/AS, par)
        curvature, bending_tangent, bending_energy = self.steel(moment, par)
        gradient = -axial_strain[..., 0]-e*curvature
        derivative = axial_tangent[..., 0]/AS+e.square()*bending_tangent
        return AS*axial_energy[..., 0]+bending_energy, curvature, gradient, derivative


class ConditionedCoupled(ConditionedBatch):
    def __init__(self, names, loads, compliances, separate=False, subdivisions=16, section_order=4):
        batch_class = SeparateBatch if separate else CoupledBatch
        self.elastic = batch_class(names, [1.], subdivisions, section_order)
        self.response = batch_class(names, loads, subdivisions, section_order)
        self.observed = torch.tensor([compliances[n] for n in names])
        self.repeat = len(loads)

    def strain(self, theta, distances):
        self.predict(theta)
        par = torch.exp(torch.cat((torch.zeros(1), theta)))
        b = self.response
        force = b.equilibrium(par)
        force_rows = []
        for i, positions in enumerate(distances):
            nodes = torch.cat((torch.zeros(1), b.g['length'][i].cumsum(0)))
            forces = torch.cat((force[i], torch.zeros(1)))
            row = []
            for x in positions:
                left = int(torch.searchsorted(nodes, torch.tensor(x)).item())-1
                ratio = (x-nodes[left])/(nodes[left+1]-nodes[left])
                n = forces[left]*(1-ratio)+forces[left+1]*ratio
                row.append(n)
            force_rows.append(torch.stack(row))
        n = torch.stack(force_rows)
        x = torch.tensor(distances)
        e = b.g['eccentricity'][:, None]
        moment = b.p[:, None]*torch.minimum(torch.full_like(x,300.),(900-x)/2)-e*n
        curvature = b.section_response(n,moment,e,par)[1]
        thickness = 3.175*b.g['thickness_ratio'][:, None]
        return 1e6*(n/(EF*101.6*thickness)+thickness/2*curvature)


def fiber_quadrature(order):
    points, weights = np.polynomial.legendre.leggauss(order)
    y, area = [], []
    for lo,hi,width in ((-H/2,-H/2+TF,B),(-H/2+TF,H/2-TF,TW),(H/2-TF,H/2,B)):
        y.extend(lo+(points+1)*(hi-lo)/2)
        area.extend(weights*(hi-lo)/2*width)
    return torch.tensor(y)/(H/2),torch.tensor(area)/AS


class FiberBatch(CoupledBatch):
    def __init__(self, names, loads, subdivisions=16, fiber_order=8):
        super().__init__(names,loads,subdivisions)
        self.r,self.w = fiber_quadrature(fiber_order)

    def section_state(self, axial_force, moment, par):
        c = self.elastic_scale[:,None] if self.elastic_scale is not None else par[0]
        if par[1].item()<1e-20:
            energy = c/(2*ES)*(axial_force.square()/AS+moment.square()/IS)
            scale = c/(ES*AS)
            one,zero = torch.ones_like(moment),torch.zeros_like(moment)
            return energy,c*axial_force/(ES*AS),c*moment/(ES*IS),scale,one,zero,one*(AS*(H/2)**2/IS)
        a, exponent = par[1:3]
        r,w = self.r,self.w
        t0,t1 = axial_force/(FY*AS),moment/(FY*AS*H/2)

        def newton_state(s):
            strain = s*(1+a*s.abs().pow(exponent-1))
            flexibility = 1+a*exponent*s.abs().pow(exponent-1)
            inverse = 1/flexibility
            k00 = torch.sum(w*inverse,dim=-1)
            k01 = torch.sum(w*r*inverse,dim=-1)
            k11 = torch.sum(w*r.square()*inverse,dim=-1)
            rhs0 = torch.sum(w*strain*inverse,dim=-1)+t0-torch.sum(w*s,dim=-1)
            rhs1 = torch.sum(w*r*strain*inverse,dim=-1)+t1-torch.sum(w*r*s,dim=-1)
            det = k00*k11-k01.square()
            z0 = (k11*rhs0-k01*rhs1)/det
            z1 = (k00*rhs1-k01*rhs0)/det
            step = (z0[...,None]+z1[...,None]*r-strain)/flexibility
            return step,strain,k00,k01,k11

        def density(s):
            return s.square()/2+a*s.abs().pow(exponent+1)/(exponent+1)

        with torch.no_grad():
            s = t0[...,None]+t1[...,None]*r/torch.sum(w*r.square())
            for _ in range(60):
                step,strain,_,_,_ = newton_state(s)
                if step.abs().max().item()<1e-11:
                    break
                energy = torch.sum(w*density(s),dim=-1)
                directional = torch.sum(w*strain*step,dim=-1)
                alpha = torch.ones_like(energy)
                for _ in range(24):
                    trial = s+alpha[...,None]*step
                    bad = torch.sum(w*density(trial),dim=-1)>energy+1e-4*alpha*directional+1e-14
                    if not bad.any():
                        break
                    alpha[bad] *= .5
                s = s+alpha[...,None]*step
            else:
                raise RuntimeError('Fiber section equilibrium did not converge')
        if par.requires_grad or axial_force.requires_grad or moment.requires_grad:
            s = s+newton_state(s)[0]
        _,strain,k00,k01,k11 = newton_state(s)
        z0 = torch.sum(w*strain,dim=-1)
        z1 = torch.sum(w*r*strain,dim=-1)/torch.sum(w*r.square())
        det = k00*k11-k01.square()
        beta = c*FY/ES
        energy = c*FY**2*AS/ES*torch.sum(w*density(s),dim=-1)
        return energy,beta*z0,beta*z1/(H/2),beta/(FY*AS),k11/det,-k01/det,k00/det

    def section_response(self, n, moment, e, par):
        energy,axial_strain,curvature,scale,l00,l01,l11 = self.section_state(-n,moment,par)
        h = H/2
        gradient = -axial_strain-e*curvature
        derivative = scale*(l00+2*(e/h)*l01+(e/h).square()*l11)
        return energy,curvature,gradient,derivative

    def steel(self, moment, par, x=None):
        energy,_,curvature,scale,_,_,l11 = self.section_state(torch.zeros_like(moment),moment,par)
        return curvature,scale*l11/(H/2)**2,energy


class ConditionedFiber(ConditionedCoupled):
    def __init__(self,names,loads,compliances,subdivisions=16,fiber_order=8):
        self.elastic = FiberBatch(names,[1.],subdivisions,fiber_order)
        self.response = FiberBatch(names,loads,subdivisions,fiber_order)
        self.observed = torch.tensor([compliances[n] for n in names])
        self.repeat = len(loads)


def activation(z):
    """Squared softplus and its first two derivatives."""
    softplus = torch.logaddexp(z, torch.zeros_like(z))
    sigmoid = torch.sigmoid(z)
    return (softplus.square() / 2, softplus * sigmoid,
            sigmoid.square() + softplus * sigmoid * (1 - sigmoid))


def neural_shape(r, parameters):
    """Return Phi', Phi'' and Phi, with Phi(0)=Phi'(0)=0, Phi''(0)=1.

    Positive output weights and convex activations guarantee a positive
    tangent. Paired affine inputs impose an even potential. The positive
    normalization separates the learned nonlinear shape from the observed
    initial compliance. Parameters contain weights, slopes, then thresholds.
    """
    weights, slopes, thresholds = parameters.split(WIDTH)
    positive = activation(slopes * (r[..., None] - thresholds))
    negative = activation(slopes * (-r[..., None] - thresholds))
    origin = activation(-slopes * thresholds)
    normalization = 1 + (2 * weights * origin[2]).sum()
    energy = (r.square() / 2 + (
        weights / slopes.square() * (positive[0] + negative[0] - 2 * origin[0])
    ).sum(dim=-1)) / normalization
    strain = (r + (weights / slopes * (positive[1] - negative[1])).sum(dim=-1)) / normalization
    tangent = (1 + (weights * (positive[2] + negative[2])).sum(dim=-1)) / normalization
    return strain, tangent, energy


class NeuralSectionBatch(CoupledBatch):
    def material(self, stress, par):
        c = self.elastic_scale[:, None, None] if self.elastic_scale is not None else par[0]
        strain, tangent, energy = neural_shape(stress / FY, par[1:-2])
        return c * FY / ES * strain, c / ES * tangent, c * FY**2 / ES * energy

    def connector(self, q, par):
        # The inherited connector takes its two positive scales in slots 3:5.
        return super().connector(q, torch.cat((par.new_zeros(3), par[-2:])))


class ConditionedNeural(ConditionedCoupled):
    def __init__(self, names, loads, compliances, subdivisions=16):
        self.elastic = CoupledBatch(names, [1.], subdivisions)
        self.response = NeuralSectionBatch(names, loads, subdivisions)
        self.observed = torch.tensor([compliances[n] for n in names])
        self.repeat = len(loads)

    def predict(self, theta):
        # Use the same elastic reference and conditioning iteration as M_B.
        linear = torch.cat((theta.new_tensor([0., -60., np.log(2.)]), theta[-2:]))
        c = self.observed * EI / 103500000.
        for _ in range(9):
            self.elastic.elastic_scale = c
            value = self.elastic.predict(linear)[:, 0]
            c = c * self.observed / value
        self.response.elastic_scale = c.repeat_interleave(self.repeat)
        return self.response.predict(torch.cat((theta.new_zeros(1), theta)))


class JointCalibration:
    def __init__(self, inputs, subdivisions=16):
        self.names = sorted(set(inputs) - set(HELDOUT))
        training = {name: inputs[name] for name in self.names}
        compliance = observed_compliances(training)
        self.batch = ConditionedNeural(self.names, TRAIN_LOADS, compliance, subdivisions)
        self.gauge = ConditionedNeural(['U90S100'], [310.], compliance, subdivisions)
        self.target = torch.tensor([training[n]['equal_force_mean'] for n in self.names])
        self.strain_target = torch.tensor(load_strains('U90S100')['microstrain'])

    def __call__(self, theta):
        displacement = ((self.batch.predict(theta) - self.target) / 10).square().mean()
        strain = ((self.gauge.strain(theta, [(100., 400.)])[0]
                   - self.strain_target) / 2000).square().mean()
        return displacement + strain


def fit_power(names, inputs, compliance, joint, previous=None, separate=False, initial=None, batch_factory=None):
    make_batch = batch_factory or (lambda names,loads,c:ConditionedCoupled(names,loads,c,separate))
    batch = make_batch(names, TRAIN_LOADS, compliance)
    gauge = make_batch(['U90S100'], [310.], compliance)
    target = torch.tensor([inputs[n]['equal_force_mean'] for n in names])
    if joint and 'U90S100' not in names:
        raise ValueError('The strain calibration design must be in training')
    def loss_value(p):
        loss = ((batch.predict(p)-target)/10).square().mean()
        if joint:
            loss += ((gauge.strain(p, [(100., 400.)])[0]-torch.tensor(load_strains('U90S100')['microstrain']))/2000).square().mean()
        return loss
    def objective(theta):
        p = torch.tensor(theta, requires_grad=True)
        loss = loss_value(p)
        loss.backward()
        return loss.item(), p.grad.numpy()
    start = np.log([.4, 12., .5, 2.]) if initial is None else np.asarray(initial)
    bounds = np.log([[1e-4, 10.], [2., 40.], [.1, 10.], [.1, 10.]])
    if previous is None:
        solved = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 30})
    else:
        solved = minimize(lambda t: loss_value(torch.tensor(t)).item(), previous['log_parameters'],
                          method='Powell', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-9, 'xtol': 1e-6})
        if solved.fun>previous['objective']:
            return previous
    return {'log_parameters': solved.x.tolist(), 'parameters': np.exp(solved.x).tolist(),
            'training': names, 'objective': float(solved.fun), 'success': bool(solved.success),
            'message': str(solved.message), 'training_strain_microstrain':
            gauge.strain(torch.tensor(solved.x), [(100., 400.)])[0].detach().tolist(),
            'optimizer': 'L-BFGS-B' if previous is None else 'Powell',
            'function_evaluations': int(solved.nfev), 'iterations': int(solved.nit),
            'initial_log_parameters': start.tolist(),
            'initial_objective': None if previous is None else previous['objective']}


def initial_neural_parameters(baseline):
    """Approximate the training-fitted M_B shape before observational fitting.

    The initialization interval uses only the maximum training force and
    unreinforced-section stress scale. No held-out response is evaluated.
    """
    a, power, slip_scale, force_scale = baseline['parameters']
    upper = float(TRAIN_LOADS.max()) * 300 * (H / 2) / IS / FY
    r = torch.linspace(0., upper, 121)
    target_strain = r * (1 + a * r.pow(power - 1))
    target_tangent = 1 + a * power * r.pow(power - 1)

    def residual(theta):
        with torch.no_grad():
            strain, tangent, _ = neural_shape(r, torch.exp(torch.tensor(theta)))
            return torch.cat(((strain - target_strain) / (1 + target_strain),
                              (tangent - target_tangent) / target_tangent)).numpy()

    seed = np.log([.3, 2., 12., 50., 8., 12., 20., 25., .65, .9, 1.05, 1.18])
    initialized = least_squares(residual, seed, bounds=BOUNDS[:-2].T,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
    theta = np.r_[initialized.x, np.log([slip_scale, force_scale])]
    return theta
