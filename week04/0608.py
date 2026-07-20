import numpy as np
import matplotlib.pyplot as plt
import math
import os

os.makedirs('figures', exist_ok=True)
plt.rcParams['font.size'] = 12

# ========== 1. 基本初等函数 ==========
x = np.linspace(-2*np.pi, 2*np.pi, 500)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(x, y1, 'b-', lw=2)
axes[0].axhline(0, color='gray', ls='--', alpha=0.5)
axes[0].set_title(r'$y = \sin(x)$', fontsize=14)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].grid(alpha=0.3)

axes[1].plot(x, y2, 'g-', lw=2)
axes[1].axhline(0, color='gray', ls='--', alpha=0.5)
axes[1].set_title(r'$y = \cos(x)$', fontsize=14)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].grid(alpha=0.3)

# tan(x) 避开奇点
xt = np.linspace(-2*np.pi, 2*np.pi, 2000)
yt = np.tan(xt)
yt[np.abs(yt) > 10] = np.nan
axes[2].plot(xt, yt, 'r-', lw=2)
axes[2].axhline(0, color='gray', ls='--', alpha=0.5)
axes[2].set_ylim(-10, 10)
axes[2].set_title(r'$y = \tan(x)$', fontsize=14)
axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.suptitle('Basic Elementary Functions', fontsize=16, y=1.02)
plt.savefig('figures/01_basic_functions.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 2. 指数、对数与幂函数 ==========
x1 = np.linspace(0.1, 4, 400)
x2 = np.linspace(-2, 2, 400)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(x1, np.exp(x1), 'b-', lw=2, label=r'$e^x$')
axes[0].plot(x1, 2**x1, 'g--', lw=2, label=r'$2^x$')
axes[0].axhline(0, color='gray', ls='--', alpha=0.5)
axes[0].set_title('Exponential Functions', fontsize=14)
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(x1, np.log(x1), 'b-', lw=2, label=r'$\ln(x)$')
axes[1].plot(x1, np.log10(x1), 'g--', lw=2, label=r'$\log_{10}(x)$')
axes[1].axhline(0, color='gray', ls='--', alpha=0.5)
axes[1].axvline(0, color='gray', ls='--', alpha=0.5)
axes[1].set_title('Logarithmic Functions', fontsize=14)
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(x2, x2**2, 'b-', lw=2, label=r'$x^2$')
axes[2].plot(x2, x2**3, 'g-', lw=2, label=r'$x^3$')
axes[2].plot(x2[x2>=0], np.sqrt(x2[x2>=0]), 'r-', lw=2, label=r'$\sqrt{x}$')
axes[2].axhline(0, color='gray', ls='--', alpha=0.5)
axes[2].axvline(0, color='gray', ls='--', alpha=0.5)
axes[2].set_title('Power Functions', fontsize=14)
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.suptitle('Exponential / Logarithmic / Power', fontsize=16, y=1.02)
plt.savefig('figures/02_exp_log_power.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 3. 极限 ==========
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# sin(x)/x
x = np.linspace(-10, 10, 1000)
y = np.sin(x) / x
y[np.isnan(y)] = 1
axes[0].plot(x, y, 'b-', lw=2)
axes[0].plot(0, 1, 'ro', ms=8, label=r'$\lim_{x\to0} \frac{\sin x}{x}=1$')
axes[0].axhline(0, color='gray', ls='--', alpha=0.5)
axes[0].axvline(0, color='gray', ls='--', alpha=0.5)
axes[0].set_title(r'$\lim_{x\to0} \frac{\sin x}{x}=1$', fontsize=14)
axes[0].legend(fontsize=12); axes[0].grid(alpha=0.3)

# (1+1/x)^x → e
x = np.linspace(1, 30, 1000)
y = (1 + 1/x)**x
axes[1].plot(x, y, 'g-', lw=2, label=r'$(1+\frac{1}{x})^x$')
axes[1].axhline(np.e, color='red', ls='--', lw=2, label=r'$e$')
axes[1].set_title(r'$\lim_{x\to\infty} (1+\frac{1}{x})^x = e$', fontsize=14)
axes[1].legend(fontsize=12); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.suptitle('Important Limits', fontsize=16, y=1.02)
plt.savefig('figures/03_limits.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 4. 导数与切线 ==========
def f(x): return x**2
def df(x): return 2*x

x = np.linspace(-3, 3, 400)
y = f(x)

x0 = 1.5
y0 = f(x0)
slope = df(x0)
tangent = y0 + slope * (x - x0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'b-', lw=3, label=r'$f(x)=x^2$')
ax.plot(x, tangent, 'r--', lw=2, label=f'Tangent at x={x0}')
ax.plot(x0, y0, 'ro', ms=10, label=f'Point ({x0}, {y0})')
ax.axhline(0, color='gray', ls='--', alpha=0.4)
ax.axvline(0, color='gray', ls='--', alpha=0.4)
ax.set_title(f'Derivative: slope of tangent = {slope}', fontsize=14)
ax.legend(fontsize=12); ax.grid(alpha=0.3)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.tight_layout()
plt.savefig('figures/04_derivative.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 5. 定积分 ==========
def f(x): return np.sin(x) + 1

a, b = 0, np.pi
x = np.linspace(a, b, 400)
y = f(x)
x_fill = np.linspace(a, b, 100)
y_fill = f(x_fill)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'b-', lw=3, label=r'$f(x)=\sin(x)+1$')
ax.fill_between(x_fill, y_fill, alpha=0.3, color='orange', label='Area = integral')
ax.axhline(0, color='gray', ls='--', alpha=0.4)
ax.set_title(r'Definite Integral: $\int_0^{\pi} (\sin(x)+1)\,dx$', fontsize=14)
ax.legend(fontsize=12); ax.grid(alpha=0.3)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.tight_layout()
plt.savefig('figures/05_definite_integral.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 6. 泰勒展开 ==========
def f(x): return np.sin(x)

x = np.linspace(-2*np.pi, 2*np.pi, 500)
y_true = f(x)

def taylor_sin(x, n):
    s = np.zeros_like(x)
    for k in range(n+1):
        s += ((-1)**k) * (x**(2*k+1)) / math.factorial(2*k+1)
    return s

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y_true, 'k-', lw=3, label=r'$\sin(x)$ exact')

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
orders = [1, 3, 5, 7, 9]
for n, c in zip(orders, colors):
    ax.plot(x, taylor_sin(x, n), '--', lw=1.5, label=f'Taylor n={n}', color=c)

ax.set_ylim(-2, 2)
ax.axhline(0, color='gray', ls='--', alpha=0.4)
ax.set_title("Taylor Series Approximation of sin(x)", fontsize=14)
ax.legend(fontsize=10, ncol=2); ax.grid(alpha=0.3)
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.tight_layout()
plt.savefig('figures/06_taylor.png', dpi=300, bbox_inches='tight')
plt.close()

# ========== 7. 3D 曲面 ==========
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(np.linspace(-3, 3, 40), np.linspace(-3, 3, 40))
Z = np.sin(np.sqrt(X**2 + Y**2))
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax1.set_title(r'$f(x,y)=\sin(\sqrt{x^2+y^2})$', fontsize=12)
fig.colorbar(surf, ax=ax1, shrink=0.5)

ax2 = fig.add_subplot(122, projection='3d')
Z2 = X**2 - Y**2
surf2 = ax2.plot_surface(X, Y, Z2, cmap='coolwarm', edgecolor='none')
ax2.set_title(r'$f(x,y)=x^2 - y^2$ (saddle)', fontsize=12)
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.suptitle('3D Surfaces of Multivariable Functions', fontsize=16, y=1.02)
plt.savefig('figures/07_3d_surfaces.png', dpi=300, bbox_inches='tight')
plt.close()


# ==============================================================
#  1. 二重积分 —— 体积 = ∬_D f(x,y) dA
# ==============================================================
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
Z = 4 - X**2 - Y**2   # 曲顶柱体
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.contour(X, Y, Z, zdir='z', offset=0, levels=10, cmap='viridis', alpha=0.6)
ax1.set_zlim(0, 4.5)
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.set_title(r'Double Integral: $\iint_D (4-x^2-y^2)\,dA$', fontsize=12)

# 底面投影区域 (圆形区域 x^2 + y^2 <= 4)
theta = np.linspace(0, 2*np.pi, 100)
cx, cy = 2*np.cos(theta), 2*np.sin(theta)
ax1.plot(cx, cy, 'r--', lw=2, label='Projection: $x^2+y^2 \\leq 4$')
ax1.legend(loc='upper right')

# 俯视图：彩色填充显示
ax2 = fig.add_subplot(122)
r = np.linspace(0, 2, 100)
t = np.linspace(0, 2*np.pi, 100)
R, T = np.meshgrid(r, t)
Xp = R * np.cos(T)
Yp = R * np.sin(T)
Zp = 4 - Xp**2 - Yp**2
cont = ax2.contourf(Xp, Yp, Zp, levels=20, cmap='viridis')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('Top View (filled contour)', fontsize=12)
ax2.set_aspect('equal')
plt.colorbar(cont, ax=ax2, label='z value')
plt.suptitle('Double Integral - Volume under Surface', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/08_double_integral.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================
#  2. 三重积分 —— 三维区域可视化
# ==============================================================
fig = plt.figure(figsize=(14, 6))

# 半球体 + 圆柱 示例：球坐标区域
ax1 = fig.add_subplot(121, projection='3d')
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi/2, 15)      # 上半球
U, V = np.meshgrid(u, v)
Xs = np.sin(V) * np.cos(U)
Ys = np.sin(V) * np.sin(U)
Zs = np.cos(V)
ax1.plot_surface(Xs, Ys, Zs, color='steelblue', alpha=0.6, edgecolor='gray', linewidth=0.3)
ax1.set_title('Sphere (upper half)\n$\\iiint_V dV$', fontsize=12)
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.set_box_aspect([1,1,1])

# 长方体区域
ax2 = fig.add_subplot(122, projection='3d')
x = np.array([0, 2, 2, 0, 0, 2, 2, 0])
y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
z = np.array([0, 0, 0, 0, 3, 3, 3, 3])
# 绘制长方体边框
for i, j, k in [(0,1,2),(0,2,3),(0,1,5),(0,3,7),(0,4,7),(0,4,5),
                (1,5,6),(1,2,6),(2,6,7),(3,7,6),(4,5,6),(4,6,7)]:
    # 只画边
    pass
# 改用简单方式画长方体
Xb, Yb = np.meshgrid([0, 2], [0, 1])
Zb_bottom = np.zeros_like(Xb)
Zb_top = np.full_like(Xb, 3)
ax2.plot_surface(Xb, Yb, Zb_bottom, alpha=0.2, color='orange')
ax2.plot_surface(Xb, Yb, Zb_top, alpha=0.2, color='orange')
Xb2, Zb2 = np.meshgrid([0, 2], [0, 3])
Yb2_front = np.zeros_like(Xb2)
Yb2_back = np.ones_like(Xb2)
ax2.plot_surface(Xb2, Yb2_front, Zb2, alpha=0.2, color='orange')
ax2.plot_surface(Xb2, Yb2_back, Zb2, alpha=0.2, color='orange')
Yb3, Zb3 = np.meshgrid([0, 1], [0, 3])
Xb3_left = np.zeros_like(Yb3)
Xb3_right = np.full_like(Yb3, 2)
ax2.plot_surface(Xb3_left, Yb3, Zb3, alpha=0.2, color='orange')
ax2.plot_surface(Xb3_right, Yb3, Zb3, alpha=0.2, color='orange')

ax2.set_title('Cuboid: $0\\leq x\\leq 2, 0\\leq y\\leq 1, 0\\leq z\\leq 3$\n$\\iiint_V f(x,y,z)\\,dV$', fontsize=12)
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.set_box_aspect([2,1,3])
plt.suptitle('Triple Integral - 3D Regions', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/09_triple_integral.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================
#  3. 曲线积分 —— ∫_C F·dr (第二类曲线积分)
# ==============================================================
fig = plt.figure(figsize=(16, 6))

# 3a. 第一类曲线积分 —— 弧长/质量
ax = fig.add_subplot(121, projection='3d')
t = np.linspace(0, 2*np.pi, 500)
xc = np.cos(t)
yc = np.sin(t)
zc = t / (2*np.pi) * 3          # 螺旋线上升
ax.plot(xc, yc, zc, 'b-', lw=3, label='Curve $C$: helix')

# 沿曲线标注方向箭头
for frac in [0.2, 0.5, 0.8]:
    idx = int(frac * len(t))
    dx = -np.sin(t[idx])
    dy = np.cos(t[idx])
    dz = 1/(2*np.pi)*3
    ax.quiver(xc[idx], yc[idx], zc[idx], dx, dy, dz,
              length=0.3, color='red', normalize=True)

ax.set_title('Line Integral (1st kind): $\\int_C f(x,y,z)\\,ds$\nArc length of helix', fontsize=12)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.legend()
ax.set_box_aspect([1,1,1.5])

# 3b. 第二类曲线积分 —— 做功
ax2 = fig.add_subplot(122)
# 向量场 F(x,y) = (-y, x) 沿圆环的功
t = np.linspace(0, 2*np.pi, 30)
xc = np.cos(t)
yc = np.sin(t)
# 场向量
Fx = -yc
Fy = xc
ax2.quiver(xc, yc, Fx, Fy, alpha=0.6, color='steelblue', scale=1.5, width=0.005,
           label='Vector field $\\mathbf{F}=(-y, x)$')
# 路径
t_dense = np.linspace(0, 2*np.pi, 200)
ax2.plot(np.cos(t_dense), np.sin(t_dense), 'r-', lw=3, label='Path $C$: unit circle')
ax2.set_title('Line Integral (2nd kind): $\\int_C \\mathbf{F}\\cdot d\\mathbf{r}$\nWork along a circle', fontsize=12)
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_aspect('equal')
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle('Line Integrals', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/10_line_integral.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================
#  4. 曲面积分 —— ∬_S F·n dS (通量)
# ==============================================================
fig = plt.figure(figsize=(16, 6))

# 4a. 第一类曲面积分 —— 曲面面积
ax1 = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
Z = X**2 + Y**2   # 抛物面
surf = ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none')
ax1.set_title('Surface Integral (1st kind): $\\iint_S f(x,y,z)\\,dS$\nParaboloid $z=x^2+y^2$', fontsize=11)
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# 4b. 第二类曲面积分 —— 通量 (向量场穿过曲面)
ax2 = fig.add_subplot(122, projection='3d')
X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 12), np.linspace(-1.5, 1.5, 12))
Z = X**2 + Y**2

# 绘制曲面
ax2.plot_surface(X, Y, Z, cmap='YlOrRd', alpha=0.5, edgecolor='gray', linewidth=0.3)

# 法向量 (近似向上)
Ux = -2*X / np.sqrt(4*X**2 + 4*Y**2 + 1)
Uy = -2*Y / np.sqrt(4*X**2 + 4*Y**2 + 1)
Uz =  1  / np.sqrt(4*X**2 + 4*Y**2 + 1)
# 每隔一个点画法向量
step = 2
ax2.quiver(X[::step,::step], Y[::step,::step], Z[::step,::step],
           Ux[::step,::step], Uy[::step,::step], Uz[::step,::step],
           length=0.4, color='darkred', alpha=0.8, normalize=True,
           label='Normal vectors $\\mathbf{n}$')

ax2.set_title('Surface Integral (2nd kind): $\\iint_S \\mathbf{F}\\cdot\\mathbf{n}\\,dS$\nFlux through surface', fontsize=11)
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.legend()
ax2.view_init(elev=30, azim=-60)

plt.suptitle('Surface Integrals', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/11_surface_integral.png', dpi=300, bbox_inches='tight')
plt.close()


# ==============================================================
#  格林公式 Green's Theorem
#  ∮_C (L dx + M dy) = ∬_D (∂M/∂x - ∂L/∂y) dA
# ==============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：闭曲线与内部区域
t = np.linspace(0, 2*np.pi, 400)
# 用一个非对称闭曲线：椭圆 + 扰动
rx, ry = 2.5, 1.8
xc = rx * np.cos(t) + 0.3 * np.cos(3*t)
yc = ry * np.sin(t) + 0.2 * np.sin(2*t)

axes[0].plot(xc, yc, 'b-', lw=3, label='Closed curve $C$')
axes[0].fill(xc, yc, alpha=0.2, color='steelblue', label='Region $D$')
axes[0].plot(xc[0], yc[0], 'ro', ms=8, label='Start/End')
# 标注方向
for frac in [0.15, 0.40, 0.65, 0.90]:
    idx = int(frac * len(t))
    dx = xc[(idx+1)%len(t)] - xc[idx]
    dy = yc[(idx+1)%len(t)] - yc[idx]
    d = np.sqrt(dx**2 + dy**2)
    axes[0].arrow(xc[idx], yc[idx], dx/d*0.3, dy/d*0.3,
                  head_width=0.15, head_length=0.15, fc='green', ec='green')
axes[0].set_title("Green's Theorem:\n$\\oint_C (L\\,dx + M\\,dy) = \\iint_D (\\frac{\\partial M}{\\partial x} - \\frac{\\partial L}{\\partial y})\\,dA$",
                  fontsize=12)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_aspect('equal')
axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)

# 右图：向量场与环量
t = np.linspace(0, 2*np.pi, 25)
xc_q = rx * np.cos(t) + 0.3 * np.cos(3*t)
yc_q = ry * np.sin(t) + 0.2 * np.sin(2*t)

# 向量场 F = (L, M) = (sin y, cos x) 用于演示
Xg, Yg = np.meshgrid(np.linspace(-3, 3, 12), np.linspace(-2.5, 2.5, 10))
L = np.sin(Yg)
M = np.cos(Xg)
axes[1].quiver(Xg, Yg, L, M, alpha=0.6, color='steelblue', scale=3, width=0.003,
               label='Vector field $\\mathbf{F}=(\\sin y,\\cos x)$')
axes[1].plot(xc, yc, 'r-', lw=2, label='Closed curve $C$')
axes[1].set_title("Circulation of $\\mathbf{F}$ around $C$\n(counterclockwise)", fontsize=12)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].set_aspect('equal')
axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

plt.suptitle('Green\'s Theorem', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('figures/12_green.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================
#  高斯公式 (散度定理) Divergence Theorem
#  ∯_S F·n dS = ∭_V div F dV
# ==============================================================
fig = plt.figure(figsize=(16, 6))

# 左图：封闭曲面 + 外法向量
ax1 = fig.add_subplot(121, projection='3d')
u = np.linspace(0, 2*np.pi, 25)
v = np.linspace(0, np.pi, 20)
U, V = np.meshgrid(u, v)
# 椭球体
a, b, c = 2.0, 1.5, 1.0
Xs = a * np.sin(V) * np.cos(U)
Ys = b * np.sin(V) * np.sin(U)
Zs = c * np.cos(V)
ax1.plot_surface(Xs, Ys, Zs, color='steelblue', alpha=0.4, edgecolor='gray', linewidth=0.3)

# 外法向量 (每隔一些点)
step = 3
Nx = np.sin(V)[::step,::step] * np.cos(U)[::step,::step] / a
Ny = np.sin(V)[::step,::step] * np.sin(U)[::step,::step] / b
Nz = np.cos(V)[::step,::step] / c
norm = np.sqrt(Nx**2 + Ny**2 + Nz**2)
ax1.quiver(Xs[::step,::step], Ys[::step,::step], Zs[::step,::step],
           Nx/norm, Ny/norm, Nz/norm,
           length=0.4, color='darkred', alpha=0.7, normalize=True,
           label='Outward normals $\\mathbf{n}$')
ax1.set_title("Divergence Theorem (Gauss):\n$\\oiint_S \\mathbf{F}\\cdot\\mathbf{n}\\,dS = \\iiint_V \\nabla\\cdot\\mathbf{F}\\,dV$",
              fontsize=11)
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.legend(fontsize=9)
ax1.set_box_aspect([a, b, c])

# 右图：散度场的体渲染示意（用颜色表示 div F）
ax2 = fig.add_subplot(122, projection='3d')
# 假设 F = (x, y, z) => div F = 3
X, Y, Z = np.meshgrid(np.linspace(-1.5, 1.5, 10),
                       np.linspace(-1.2, 1.2, 8),
                       np.linspace(-0.8, 0.8, 6))

divF = np.full_like(X, 3)  # div F = 3 处处相等

# 用散点大小和颜色表示散度值
sc = ax2.scatter(X.ravel(), Y.ravel(), Z.ravel(),
                 c=divF.ravel(), cmap='Reds', s=divF.ravel()*30,
                 alpha=0.6, label='$\\nabla\\cdot\\mathbf{F}=3$')

# 画一个半透明外壳示意闭合曲面
u_d = np.linspace(0, 2*np.pi, 30)
v_d = np.linspace(0, np.pi, 20)
U_d, V_d = np.meshgrid(u_d, v_d)
Xs_d = 1.5 * np.sin(V_d) * np.cos(U_d)
Ys_d = 1.2 * np.sin(V_d) * np.sin(U_d)
Zs_d = 0.8 * np.cos(V_d)
ax2.plot_wireframe(Xs_d, Ys_d, Zs_d, color='gray', alpha=0.2, linewidth=0.5)

ax2.set_title("Volume with constant divergence\n$\\mathbf{F}=(x,y,z) \\Rightarrow \\nabla\\cdot\\mathbf{F}=3$",
              fontsize=11)
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.legend(fontsize=9)
plt.colorbar(sc, ax=ax2, shrink=0.5, label='div F')

plt.suptitle("Gauss's Divergence Theorem", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('figures/13_gauss.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================
#  斯托克斯公式 Stokes' Theorem
#  ∮_C F·dr = ∬_S (∇×F)·n dS
# ==============================================================
fig = plt.figure(figsize=(16, 6))

ax = fig.add_subplot(121, projection='3d')
# 曲面：部分抛物面 z = 4 - x^2 - y^2
X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 30), np.linspace(-1.5, 1.5, 30))
Z = 4 - X**2 - Y**2
mask = X**2 + Y**2 <= 2.0
Z_masked = np.where(mask, Z, np.nan)
surf = ax.plot_surface(X, Y, Z_masked, cmap='coolwarm', alpha=0.7, edgecolor='none')
ax.set_title("Stokes' Theorem:\n$\\oint_C \\mathbf{F}\\cdot d\\mathbf{r} = \\iint_S (\\nabla\\times\\mathbf{F})\\cdot\\mathbf{n}\\,dS$",
             fontsize=11)

# 边界曲线 C (在曲面边缘)
theta = np.linspace(0, 2*np.pi, 100)
r = np.sqrt(2.0)
cx = r * np.cos(theta)
cy = r * np.sin(theta)
cz = 4 - cx**2 - cy**2
ax.plot(cx, cy, cz, 'r-', lw=3, label='Boundary curve $C = \\partial S$')

# 曲面上的法向量
step = 3
X_n = X[::step,::step]
Y_n = Y[::step,::step]
Z_n = Z_masked[::step,::step]  # 可能会是nan
# 只对有效区域画法向量
valid = ~np.isnan(Z_n)
Xn_v = X_n[valid]
Yn_v = Y_n[valid]
Zn_v = Z_n[valid]
# 曲面的法向量 (向上)
Nx_s = 2*Xn_v
Ny_s = 2*Yn_v
Nz_s = np.ones_like(Xn_v)
norm_s = np.sqrt(Nx_s**2 + Ny_s**2 + Nz_s**2)
ax.quiver(Xn_v, Yn_v, Zn_v,
          Nx_s/norm_s, Ny_s/norm_s, Nz_s/norm_s,
          length=0.3, color='darkgreen', alpha=0.6, normalize=True,
          label='Normals $\\mathbf{n}$ on $S$')

ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.legend(fontsize=9)
ax.view_init(elev=25, azim=-50)

# 右图：旋度（curl）示意
ax2 = fig.add_subplot(122, projection='3d')
# 向量场 F = (-y, x, 0) => curl F = (0, 0, 2) 方向向上
# 在曲面上画旋度向量
Xc, Yc = np.meshgrid(np.linspace(-1.2, 1.2, 8), np.linspace(-1.2, 1.2, 8))
mask2 = Xc**2 + Yc**2 <= 2.0
Xc_v = Xc[mask2]
Yc_v = Yc[mask2]
Zc_v = 4 - Xc_v**2 - Yc_v**2

# curl F = (0, 0, 2) 垂直向上
curl_x = np.zeros_like(Xc_v)
curl_y = np.zeros_like(Xc_v)
curl_z = np.full_like(Xc_v, 2)

ax2.quiver(Xc_v, Yc_v, Zc_v, curl_x, curl_y, curl_z,
           length=0.5, color='purple', alpha=0.7, normalize=True,
           label='$\\nabla\\times\\mathbf{F} = (0,0,2)$')

# 曲面
Xp, Yp = np.meshgrid(np.linspace(-1.5, 1.5, 20), np.linspace(-1.5, 1.5, 20))
Zp = 4 - Xp**2 - Yp**2
mask_p = Xp**2 + Yp**2 <= 2.0
Zp_m = np.where(mask_p, Zp, np.nan)
ax2.plot_surface(Xp, Yp, Zp_m, cmap='coolwarm', alpha=0.3, edgecolor='none')

# 边界曲线
ax2.plot(cx, cy, cz, 'r-', lw=2, label='Boundary $C$')

ax2.set_title("Curl field on surface\n$\\nabla\\times\\mathbf{F}$ points upward",
              fontsize=11)
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.legend(fontsize=9)
ax2.view_init(elev=25, azim=-50)

plt.suptitle("Stokes' Theorem", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('figures/14_stokes.png', dpi=300, bbox_inches='tight')
plt.close()

print("All three important formulas (Green, Gauss, Stokes) visualized!")
print("✅ All 14 figures saved to figures/ folder!")