# Stateful numerical updates

Derive an explicit per-parameter state transition before translating it into
the repository's tensor/dispatcher API. Read actual option types and helpers.
State the order of gradient transformation, moment update and parameter update.
Preserve missing-gradient behavior, device guards, counter placement, checkpoint
keys and complex representations. Every supplied behavior flag needs a defined
effect; a parameter with no direct reads can reveal an omitted branch.

For SGD, maximize negates the objective gradient before coupled weight decay.
Momentum initializes its buffer from the first transformed gradient, then uses
mu*buffer+(1-dampening)*gradient on later steps. Nesterov uses gradient+mu*buffer.
Apply a descending parameter update with the resulting direction. Confirm this
repository's initialization and sparse-gradient rules rather than assuming them.

For Adam, increment the correct step counter, update m=beta1*m+(1-beta1)*g and
v=beta2*v+(1-beta2)*g*g, and bias-correct both moments. AMSGrad retains the maximum
uncorrected v. Epsilon belongs after the square root of the bias-corrected v;
moving it changes numerical semantics. Coupled decay changes g before moments;
decoupled decay scales the parameter separately. Maximize changes the gradient
sign before these operations. Treat complex moments through the framework's real
component representation when that is its public convention.

Capturable execution must retain device-resident tensor steps and tensor-valued
hyperparameters without .item(), Python branching on device data or host sync.
Differentiable updates must preserve the needed graph values: avoid overwriting
a value later needed by autograd. Follow local helper/type conventions, including
supported devices and compiler paths. A literal recurrence is a starting point,
not proof of sparse, mixed precision, graph capture or differentiable behavior.

Implement a working candidate after bounded inspection. Use independent multi-step
numerical failures to refine it. Do not submit a copied stub or repeat unchanged
source. Establish correctness before foreach/fusion or allocation optimizations.
