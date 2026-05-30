# Model Equations

## Geometry

- `L1 = 3.261 m`: truck wheelbase.
- `L1c = 0.261 m`: hitch offset ahead of the truck rear axle.
- `L2 = 10.0 m`: trailer length from hitch to trailer rear axle.

The hitch is in front of the truck rear axle.

## Geometry Reconstruction

Given trailer pose `(X2, Y2, psi2)` and truck heading `psi1`:

```python
Xh = X2 + L2*cos(psi2)
Yh = Y2 + L2*sin(psi2)

X1 = Xh - L1c*cos(psi1)
Y1 = Yh - L1c*sin(psi1)

gamma = wrap_to_pi(psi1 - psi2)
```

## Full Plant

Full plant state:

```python
x = [X2, Y2, psi1, psi2]
```

Inputs:

```python
delta_f  # physical truck front steering angle
V1       # truck rear-axle speed
```

Equations:

```python
gamma = wrap_to_pi(psi1 - psi2)
V2 = V1 * (cos(gamma) - (L1c/L1)*sin(gamma)*tan(delta_f))
psi1_dot = (V1/L1) * tan(delta_f)
psi2_dot = (V1/L2) * (sin(gamma) + (L1c/L1)*cos(gamma)*tan(delta_f))
X2_dot = V2*cos(psi2)
Y2_dot = V2*sin(psi2)
```

Wrap `psi1` and `psi2` after integration.

## Trailer Virtual Model

Trailer LTV MPC state:

```python
x_T = [X2, Y2, psi2]
```

Input:

```python
u_T = [delta_T, V2]
```

Dynamics:

```python
X2_dot = V2*cos(psi2)
Y2_dot = V2*sin(psi2)
psi2_dot = (V2/L2)*tan(delta_T)
```

## Virtual-To-Actual Mapping

Forward:

```python
D = cos(gamma) + sin(gamma)*tan(delta_T)
N = -sin(gamma) + cos(gamma)*tan(delta_T)
```

Reverse stabilizing:

```python
D = cos(gamma) - sin(gamma)*tan(delta_T)
N = sin(gamma) - cos(gamma)*tan(delta_T)
```

Both directions:

```python
delta_f = atan2(L1*N, L1c*D)
```

Speed preserves the requested physical trailer speed after choosing the
physical steering angle:

```python
speed_gain = cos(gamma) - (L1c/L1)*sin(gamma)*tan(delta_f)
V1 = V2/speed_gain
```

The denominator `D` must stay away from zero and `delta_f` must respect the
physical steering limit.

## LTV Linearization

Linearize the virtual trailer dynamics around `(x_lin_k, u_lin_k)`:

```python
Ac = [[0, 0, -V2_lin*sin(psi2_lin)],
      [0, 0,  V2_lin*cos(psi2_lin)],
      [0, 0,  0]]

Bc = [[0,                              cos(psi2_lin)],
      [0,                              sin(psi2_lin)],
      [(V2_lin/L2)*sec(delta_T)^2,     tan(delta_T)/L2]]
```

Euler discretization:

```python
Ad = I + Ts*Ac
Bd = Ts*Bc
gd = Ts*(f(x_lin, u_lin) - Ac*x_lin - Bc*u_lin)
```
