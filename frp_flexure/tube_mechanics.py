"""Section-based tube shape, projected amplitudes and neural calibration."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from frp_flexure.data import first_crossing
from frp_flexure.tube_network import PENALTIES, SEEDS, initial_parameters, peaks, raw_network
QUERY_PAIRS = ((.3, .5), (.5, .7), (.3, .7))
NODE, WEIGHT = np.polynomial.legendre.leggauss(24)
NODE, WEIGHT = (NODE + 1) / 2, WEIGHT / 2


def displacement(parameters, traces, loads):
    p = np.asarray(loads)
    d = np.array([t.depth for t in traces])[:, None, None]
    wrap = np.array([t.wrap for t in traces])[:, None, None]
    filled = np.array([t.filled for t in traces])[:, None, None]
    steel_i = (d**4-(d-4)**4)/12
    core_i = (d-4)**4/12*filled
    frp_i = ((d+2*wrap)**4-d**4)/12
    rigidity = 200000*steel_i+25000*core_i
    transition = 634*steel_i/(d/2)*(1+(parameters[5]-1)*filled)
    eta = parameters[6]*(1-filled)+parameters[7]*filled
    frp_rigidity = 241000*frp_i*eta
    amplitude = np.exp(parameters[2]*(1-filled)+parameters[3]*filled)
    exponent = parameters[4]
    moment = p[..., None]*1000*490/4*NODE
    steel_moment = moment.copy()
    for _ in range(60):
        power = (steel_moment/transition)**(exponent-1)
        curvature = steel_moment/rigidity*(1+amplitude*power)
        derivative = (1+amplitude*exponent*power)/rigidity
        step = (steel_moment+frp_rigidity*curvature-moment)/(1+frp_rigidity*derivative)
        steel_moment -= step
        if np.max(abs(step)) < 1e-6:
            break
    else:
        raise RuntimeError('Composite section equilibrium did not converge')
    curvature = steel_moment/rigidity*(1+amplitude*(steel_moment/transition)**(exponent-1))
    scale = np.exp(parameters[0]+parameters[1]*(d[:, 0, 0]-32)/5)
    return scale[:, None]*490**2/4*np.sum(WEIGHT*NODE*curvature, axis=-1)


def physical_parameters(theta):
    return np.r_[0., 0., theta[:3], 1., theta[3:]]


def fraction(theta, traces, loads, reference_peaks):
    loads = np.asarray(loads)
    pc = np.asarray(reference_peaks)
    all_loads = np.column_stack((loads, .2*pc, .8*pc))
    u = displacement(physical_parameters(theta), traces, all_loads)
    return (u[:, 1]-u[:, 0])/(u[:, 3]-u[:, 2])


def observations(traces):
    expanded, loads, target, ids, weight = [], [], [], [], []
    for i, t in enumerate(traces):
        b, pc = t.branch, t.peak
        anchor = first_crossing(b, .2*pc)
        rows = b[(b[:, 1] > .2*pc) & (b[:, 1] < .8*pc)]
        rows = np.vstack((rows, [first_crossing(b, .8*pc), .8*pc]))
        for u, p in rows:
            expanded.append(t); loads.append([.2*pc,p]); target.append(u-anchor)
            ids.append(i); weight.append(1/np.sqrt(len(traces)*len(rows)))
    return expanded, np.array(loads), np.array(target), np.array(ids), np.array(weight)


def shape_fit(traces):
    expanded, loads, target, ids, weight = observations(traces)
    pc = np.array([t.peak for t in expanded])
    def project(theta):
        chi = fraction(theta, expanded, loads, pc)
        amplitude = np.bincount(ids, target*chi*weight**2)/np.bincount(ids, chi**2*weight**2)
        return chi, amplitude
    def residual(theta):
        chi, amp = project(theta)
        return (chi*amp[ids]-target)*weight
    solved = least_squares(residual, [-2.,-1.,5.,.3,.7],
                           bounds=([-9.,-9.,1.1,0.,0.], [3.,3.,15.,1.,1.]),
                           max_nfev=1000, ftol=1e-9, xtol=1e-9, gtol=1e-8)
    chi, amp = project(solved.x)
    amp_weight = np.sqrt(np.bincount(ids, chi**2*weight**2))/5
    return {'theta': solved.x.tolist(), 'amplitudes_mm': amp.tolist(),
            'amplitude_weights': amp_weight.tolist(), 'shape_rmse_mm': float(np.linalg.norm(residual(solved.x))),
            'success': bool(solved.success)}


def amplitude_fit(traces, shape, penalty, seed):
    features = np.array([t.features for t in traces])
    target, weight = np.array(shape['amplitudes_mm']), np.array(shape['amplitude_weights'])
    regularizer = np.sqrt(penalty)*np.eye(21)[:-1]
    def evaluate(theta):
        raw, jac = raw_network(theta, features, 1, width=4)
        value = np.logaddexp(0.,raw[:,0]); derivative = expit(raw[:,0])[:,None]*jac[:,0,:]
        return (value-target)*weight, derivative*weight[:,None]
    def residual(theta):
        return np.r_[evaluate(theta)[0],regularizer@theta]
    def jacobian(theta):
        return np.vstack((evaluate(theta)[1],regularizer))
    start = initial_parameters(3,1,seed); start[-1]=np.log(np.expm1(np.mean(target)))
    sol = least_squares(residual,start,jac=jacobian,max_nfev=800,
                        ftol=1e-8,xtol=1e-8,gtol=1e-8)
    return {'theta':sol.x.tolist(),'cost':float(sol.cost),'penalty':penalty,'seed':seed}


def predict(model, peak_model, trace, loads):
    pc = peaks(peak_model,trace.features)[0,1]
    if min(loads) < .2*pc-1e-10 or max(loads) > .8*pc+1e-10:
        return None
    amplitude = np.logaddexp(0.,raw_network(np.array(model['amplitude']['theta']),
                                           np.array([trace.features]),1,width=4)[0][0,0])
    return float(amplitude*fraction(model['shape']['theta'],[trace],np.array([loads]),[pc])[0])


def score(model, peak_model, traces):
    rows=[]
    for t in traces:
        for pair in QUERY_PAIRS[:3]:
            loads=np.array(pair)*t.peak
            value=predict(model,peak_model,t,loads)
            if value is not None:
                observed=first_crossing(t.branch,loads[1])-first_crossing(t.branch,loads[0])
                rows.append({'configuration':t.key,'loads_kn':loads.tolist(),
                             'observed_mm':observed,'predicted_mm':value,'error_mm':value-observed})
    return rows


def fit_protocol(source, by_key):
    training=[by_key[k] for k in source['training']]
    candidates={str(p):[] for p in PENALTIES}
    for fold in range(3):
        train=[t for t in training if source['folds'][t.key]!=fold]
        valid=[t for t in training if source['folds'][t.key]==fold]
        shape=shape_fit(train)
        for penalty in PENALTIES:
            trials=[amplitude_fit(train,shape,penalty,seed) for seed in SEEDS]
            model={'shape':shape,'amplitude':min(trials,key=lambda t:t['cost'])}
            rows=score(model,source['inner_peaks'][str(fold)],valid)
            candidates[str(penalty)].append({'fold':fold,'model':model,'evaluation':rows})
    scores={p:float(np.mean([r['error_mm']**2 for f in folds for r in f['evaluation']]))
            for p,folds in candidates.items()}
    chosen=float(min(scores,key=scores.get))
    shape=shape_fit(training)
    trials=[amplitude_fit(training,shape,chosen,seed) for seed in SEEDS]
    return {'shape':shape,'amplitude':min(trials,key=lambda t:t['cost']),
            'training':source['training'],'selection':scores,'selection_fits':candidates}


def common_loads(a, b):
    """Central two thirds of the observed overlapping 20--80% windows."""
    lower = max(0.2*a.peak, 0.2*b.peak)
    upper = min(0.8*a.peak, 0.8*b.peak)
    if upper <= lower:
        return None
    return np.array([lower+(upper-lower)/6, upper-(upper-lower)/6])
