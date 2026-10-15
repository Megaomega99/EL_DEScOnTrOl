// Núcleo numérico del simulador interactivo del carro-péndulo.
// Por qué existe: la charla necesita que el público pruebe los controladores en el celular,
// sin Python. Este archivo traduce a JavaScript, operación por operación, el modelo y los
// controladores de src/cartpole_lab (dinámica de Gymnasium, PID y fuzzy en cascada, LQR por
// Riccati, MPC lineal con restricciones blandas, políticas tabulares y redes exportadas).
// Se mantiene separado de la interfaz para poder verificarlo contra Python con Node
// (simulador/verify_core.py), igual que los tests del repo verifican el código Python.
const Core = (() => {
  "use strict";

  // ---------------------------------------------------------------- planta
  // Constantes de CartPole-v1 (Gymnasium 1.3.0), las mismas que lee params.py.
  const P = {
    gravity: 9.8, masscart: 1.0, masspole: 0.1, length: 0.5, forceMag: 10.0, tau: 0.02,
    xThreshold: 2.4, thetaThreshold: (12 * 2 * Math.PI) / 360,
  };
  P.totalMass = P.masspole + P.masscart;
  P.polemassLength = P.masspole * P.length;

  // s_{k+1} = s_k + τ·f(s_k, F): Euler explícito, mismo orden de operaciones que dynamics.py.
  function step(s, force) {
    const [x, xDot, th, thDot] = s;
    const c = Math.cos(th), sn = Math.sin(th);
    const temp = (force + P.polemassLength * thDot * thDot * sn) / P.totalMass;
    const thAcc = (P.gravity * sn - c * temp) /
      (P.length * (4.0 / 3.0 - (P.masspole * c * c) / P.totalMass));
    const xAcc = temp - (P.polemassLength * thAcc * c) / P.totalMass;
    return [x + P.tau * xDot, xDot + P.tau * xAcc, th + P.tau * thDot, thDot + P.tau * thAcc];
  }

  function failure(s) {
    const out = Math.abs(s[0]) > P.xThreshold, fell = Math.abs(s[2]) > P.thetaThreshold;
    return out && fell ? "ambos" : out ? "carro" : fell ? "poste" : null;
  }

  const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

  // RNG reproducible (mulberry32): el entrenamiento en el navegador debe poder repetirse.
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // Estado inicial oficial de CartPole-v1: U(−0.05, 0.05)^4.
  const initialState = (rand, h = 0.05) => [0, 0, 0, 0].map(() => (2 * rand() - 1) * h);

  // ---------------------------------------------------------------- PID en cascada (pid.py)
  const THETA_REF_LIMIT = 0.25 * P.thetaThreshold; // 3°: el lazo de posición pide como mucho esto
  function makePID(g) {
    let integral = 0;
    return {
      reset() { integral = 0; },
      act(s) {
        const thRef = clip(-(g.kp_x * s[0] + g.kd_x * s[1]), -THETA_REF_LIMIT, THETA_REF_LIMIT);
        const e = s[2] - thRef;
        const raw = g.kp_theta * e + g.ki_theta * integral + g.kd_theta * s[3];
        const f = clip(raw, -P.forceMag, P.forceMag);
        const windingUp = raw !== f && Math.sign(e) === Math.sign(raw); // anti-windup condicional
        if (!windingUp) integral += e * P.tau;
        return { force: f, info: { thetaRef: thRef, error: e, integral,
          terms: [g.kp_theta * e, g.ki_theta * integral, g.kd_theta * s[3]] } };
      },
    };
  }

  // ---------------------------------------------------------------- fuzzy Takagi-Sugeno (fuzzy.py)
  // Tabla MacVicar-Whelan: filas = error (NG..PG), columnas = velocidad angular (NG..PG).
  const RULES = [
    [-1, -1, -1, -0.5, 0], [-1, -1, -0.5, 0, 0.5], [-1, -0.5, 0, 0.5, 1],
    [-0.5, 0, 0.5, 1, 1], [0, 0.5, 1, 1, 1],
  ];
  const LABELS = ["NG", "NP", "Z", "PP", "PG"];
  // Membresías triangulares con picos en `c` (c[0] = −1, c[4] = 1). Cada triángulo baja a cero
  // en los picos vecinos, así que mover los picos interiores conserva la partición de la unidad
  // del diseño original; con picos equiespaciados se recupera exactamente memberships() de Python.
  function memberships(z, c) {
    const v = clip(z, -1, 1), mu = [0, 0, 0, 0, 0];
    for (let i = 0; i < 5; i++) {
      if (v <= c[i]) mu[i] = i === 0 ? 1 : Math.max(0, (v - c[i - 1]) / (c[i] - c[i - 1]));
      else mu[i] = i === 4 ? 1 : Math.max(0, (c[i + 1] - v) / (c[i + 1] - c[i]));
    }
    return mu;
  }
  function makeFuzzy(g, centers) {
    return {
      reset() {},
      act(s) {
        const thRef = clip(-(g.kp_x * s[0] + g.kd_x * s[1]), -THETA_REF_LIMIT, THETA_REF_LIMIT);
        const ze = (s[2] - thRef) / g.theta_scale, zd = s[3] / g.theta_dot_scale;
        const me = memberships(ze, centers.e), md = memberships(zd, centers.d);
        let num = 0, den = 0;
        const act = RULES.map((row, i) => row.map((r, j) => {
          const w = me[i] * md[j];
          num += w * r; den += w;
          return w;
        }));
        const f = clip((g.force_scale * num) / den, -P.forceMag, P.forceMag);
        return { force: f, info: { activation: act, me, md, ze, zd, thetaRef: thRef } };
      },
    };
  }

  // ---------------------------------------------------------------- álgebra lineal mínima
  const zeros = (r, c) => Array.from({ length: r }, () => new Float64Array(c));
  function matmul(A, B) {
    const n = A.length, m = B[0].length, k = B.length, C = zeros(n, m);
    for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) {
      const a = A[i][p];
      if (a !== 0) for (let j = 0; j < m; j++) C[i][j] += a * B[p][j];
    }
    return C;
  }
  const transpose = (A) => Array.from(A[0], (_, j) => Float64Array.from(A, (row) => row[j]));
  const matvec = (A, v) => A.map((row) => row.reduce((acc, a, j) => acc + a * v[j], 0));
  function cholesky(H) { // H = L Lᵀ (H simétrica definida positiva)
    const n = H.length, L = zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let sum = H[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(sum, 1e-300)) : sum / L[j][j];
    }
    return L;
  }
  function cholSolve(L, b) {
    const n = L.length, y = new Float64Array(n), x = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
    return x;
  }

  // ---------------------------------------------------------------- LQR (lqr.py)
  // Linealización de Euler alrededor del equilibrio: A = I + τ A_c, B = τ B_c (linearization.py).
  const den = P.length * (4 / 3 - P.masspole / P.totalMass);
  const a32 = P.gravity / den, b3 = -1 / (P.totalMass * den);
  const a12 = -(P.polemassLength * a32) / P.totalMass, b1 = 1 / P.totalMass - (P.polemassLength * b3) / P.totalMass;
  const A = [[1, P.tau, 0, 0], [0, 1, P.tau * a12, 0], [0, 0, 1, P.tau], [0, 0, P.tau * a32, 1]].map((row) => Float64Array.from(row));
  const B = Float64Array.from([0, P.tau * b1, 0, P.tau * b3]);

  // Pesos de Bryson: Q_ii = 1/(máximo aceptable)², R = 1/(fuerza aceptable)²; velocidades con peso 0.
  const brysonWeights = (xMax, thMax, fMax) => ({
    q: [1 / (xMax * xMax), 0, 1 / (thMax * thMax), 0], r: 1 / (fMax * fMax),
  });
  // Riccati discreta por iteración de valores; converge porque (A, B) es estabilizable y
  // (A, Q^½) detectable con estos pesos. Devuelve K (F = −K s) y P (coste terminal del MPC).
  function dlqr(q, r) {
    let Pm = zeros(4, 4);
    for (let i = 0; i < 4; i++) Pm[i][i] = q[i];
    const At = transpose(A);
    let K = new Float64Array(4);
    for (let it = 0; it < 200000; it++) {
      const PB = matvec(Pm, B), PA = matmul(Pm, A);
      const s = r + B.reduce((acc, b, i) => acc + b * PB[i], 0);
      K = Float64Array.from([0, 1, 2, 3], (j) => B.reduce((acc, b, i) => acc + b * PA[i][j], 0) / s);
      const AtPA = matmul(At, PA), AtPB = matvec(At, PB), next = zeros(4, 4);
      let diff = 0, scale = 0;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
        next[i][j] = (i === j ? q[i] : 0) + AtPA[i][j] - AtPB[i] * K[j];
        diff = Math.max(diff, Math.abs(next[i][j] - Pm[i][j]));
        scale = Math.max(scale, Math.abs(next[i][j]));
      }
      Pm = next;
      if (diff <= 1e-12 * Math.max(1, scale)) break;
    }
    for (let i = 0; i < 4; i++) for (let j = 0; j < i; j++) Pm[i][j] = Pm[j][i] = 0.5 * (Pm[i][j] + Pm[j][i]);
    return { K, P: Pm };
  }
  function makeLQR(K) {
    return {
      reset() {},
      act(s) {
        const raw = -(K[0] * s[0] + K[1] * s[1] + K[2] * s[2] + K[3] * s[3]);
        return { force: clip(raw, -P.forceMag, P.forceMag), info: { raw } };
      },
    };
  }

  // ---------------------------------------------------------------- MPC lineal (mpc.py)
  // Mismo problema que LinearMPC: coste del LQR con coste terminal P, |F| ≤ F_lim (dura) y
  // |x_i| ≤ x_lim, |θ_i| ≤ θ_lim blandas con penalización L1 de peso w (holguras σ ≥ 0).
  // CVXPY + Clarabel no existen en el navegador; se resuelve el mismo QP con un método de
  // punto interior primal-dual de Mehrotra (la familia de Clarabel), sobre la forma
  // condensada (las fuerzas son las únicas variables de decisión) y eliminando las holguras
  // por bloques, de modo que cada iteración solo factoriza una matriz N×N. Los búferes se
  // reservan una vez: en el celular se resuelve 50 veces por segundo.
  // La precisión se contrasta con CVXPY en verify_core.py.
  function makeMPC(opts) {
    const N = opts.horizon, { q, r } = opts, fLim = opts.forceLimit;
    const lims = [opts.xLimit, opts.thetaLimit], w = opts.slackWeight ?? 1e4;
    const Pt = dlqr(q, r).P;
    // Potencias de A: s_i = A^i s0 + Σ_{j<i} A^{i-1-j} B u_j.
    const Ap = [zeros(4, 4)];
    for (let i = 0; i < 4; i++) Ap[0][i][i] = 1;
    for (let i = 1; i <= N; i++) Ap.push(matmul(A, Ap[i - 1]));
    const AB = Ap.map((M) => matvec(M, B));
    // Coste condensado ½uᵀHu + gᵀu con g = F s0 (H y F en arreglos planos, fila mayor).
    const H = new Float64Array(N * N), F = new Float64Array(N * 4);
    for (let i = 1; i <= N; i++) {
      const Qi = zeros(4, 4);
      for (let k = 0; k < 4; k++) for (let c = 0; c < 4; c++) Qi[k][c] = i === N ? Pt[k][c] : k === c ? q[k] : 0;
      const G = Array.from({ length: i }, (_, j) => AB[i - 1 - j]);
      const QG = G.map((col) => matvec(Qi, col)), QA = matmul(Qi, Ap[i]);
      for (let a = 0; a < i; a++) {
        for (let b = 0; b < i; b++) { let v = 0; for (let k = 0; k < 4; k++) v += G[a][k] * QG[b][k]; H[a * N + b] += 2 * v; }
        for (let c = 0; c < 4; c++) { let v = 0; for (let k = 0; k < 4; k++) v += G[a][k] * QA[k][c]; F[a * 4 + c] += 2 * v; }
      }
    }
    for (let a = 0; a < N; a++) H[a * N + a] += 2 * r;
    // Filas de estado k = 0..2N−1 (x_1..x_N y luego θ_1..θ_N), escaladas por su límite:
    // valor/lim = a_k·u + d_k(s0); a_k solo tiene len[k] = i entradas no nulas.
    const M2 = 2 * N, Ar = new Float64Array(M2 * N), Dr = new Float64Array(M2 * 4), len = new Int32Array(M2), wS = new Float64Array(M2);
    for (let v = 0; v < 2; v++) {
      const comp = v === 0 ? 0 : 2;
      for (let i = 1; i <= N; i++) {
        const k = v * N + i - 1;
        for (let j = 0; j < i; j++) Ar[k * N + j] = AB[i - 1 - j][comp] / lims[v];
        for (let c = 0; c < 4; c++) Dr[k * 4 + c] = Ap[i][comp][c] / lims[v];
        len[k] = i; wS[k] = w * lims[v]; // σ escalada: coste w·lim·σ̃
      }
    }
    const nb = 2 * N, m = nb + 3 * M2;
    let scaleD = 1;
    for (let k = 0; k < M2; k++) scaleD = Math.max(scaleD, wS[k]);
    // Búferes reutilizados entre llamadas.
    const u = new Float64Array(N), sig = new Float64Array(M2), sl = new Float64Array(m), lam = new Float64Array(m);
    const g = new Float64Array(N), d = new Float64Array(M2), au = new Float64Array(M2);
    const rp = new Float64Array(m), rdu = new Float64Array(N), rds = new Float64Array(M2), D = new Float64Array(m);
    const Mred = new Float64Array(N * N), Tk = new Float64Array(M2), Ek = new Float64Array(M2);
    const vv = new Float64Array(m), ru = new Float64Array(N), rs = new Float64Array(M2), y = new Float64Array(N);
    const mkStep = () => ({ du: new Float64Array(N), dsig: new Float64Array(M2), dlam: new Float64Array(m), ds: new Float64Array(m) });
    const aff = mkStep(), cor = mkStep(), rcA = new Float64Array(m), rcC = new Float64Array(m), gdx = new Float64Array(m);
    const uSave = new Float64Array(N), sigSave = new Float64Array(M2);

    function choleskyInPlace(M) { // M (simétrica, plana) → factor L en el triángulo inferior
      for (let j = 0; j < N; j++) {
        let s = M[j * N + j];
        for (let k = 0; k < j; k++) s -= M[j * N + k] * M[j * N + k];
        const ljj = Math.sqrt(Math.max(s, 1e-14));
        M[j * N + j] = ljj;
        for (let i = j + 1; i < N; i++) {
          let t = M[i * N + j];
          for (let k = 0; k < j; k++) t -= M[i * N + k] * M[j * N + k];
          M[i * N + j] = t / ljj;
        }
      }
    }
    function solveL(M, b, x) { // (L Lᵀ) x = b
      for (let i = 0; i < N; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= M[i * N + k] * y[k]; y[i] = s / M[i * N + i]; }
      for (let i = N - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < N; k++) s -= M[k * N + i] * x[k]; x[i] = s / M[i * N + i]; }
    }
    function residuals() { // r_p = Gx + s − h ;  r_d = Px + q + Gᵀλ
      for (let k = 0; k < M2; k++) { let v = 0; const o = k * N; for (let j = 0; j < len[k]; j++) v += Ar[o + j] * u[j]; au[k] = v; }
      for (let i = 0; i < N; i++) { rp[2 * i] = u[i] / fLim + sl[2 * i] - 1; rp[2 * i + 1] = -u[i] / fLim + sl[2 * i + 1] - 1; }
      for (let k = 0; k < M2; k++) {
        const o = nb + 3 * k;
        rp[o] = au[k] - sig[k] + sl[o] - (1 - d[k]);
        rp[o + 1] = -au[k] - sig[k] + sl[o + 1] - (1 + d[k]);
        rp[o + 2] = -sig[k] + sl[o + 2];
      }
      for (let i = 0; i < N; i++) { let v = g[i]; for (let j = 0; j < N; j++) v += H[i * N + j] * u[j]; rdu[i] = v + (lam[2 * i] - lam[2 * i + 1]) / fLim; }
      for (let k = 0; k < M2; k++) {
        const o = nb + 3 * k, c = lam[o] - lam[o + 1], row = k * N;
        for (let j = 0; j < len[k]; j++) rdu[j] += Ar[row + j] * c;
        rds[k] = wS[k] - lam[o] - lam[o + 1] - lam[o + 2];
      }
    }
    function newton(rc, st) { // sistema de Newton con la matriz reducida ya factorizada
      for (let i = 0; i < m; i++) vv[i] = D[i] * rp[i] - rc[i] / sl[i]; // D r_p − S⁻¹ r_c
      for (let i = 0; i < N; i++) ru[i] = -rdu[i] - (vv[2 * i] - vv[2 * i + 1]) / fLim;
      for (let k = 0; k < M2; k++) {
        const o = nb + 3 * k, c = vv[o] - vv[o + 1], row = k * N;
        for (let j = 0; j < len[k]; j++) ru[j] -= Ar[row + j] * c;
        rs[k] = -rds[k] + vv[o] + vv[o + 1] + vv[o + 2];
      }
      for (let k = 0; k < M2; k++) { const c = (Ek[k] * rs[k]) / Tk[k], row = k * N; for (let j = 0; j < len[k]; j++) ru[j] -= c * Ar[row + j]; }
      solveL(Mred, ru, st.du);
      for (let i = 0; i < N; i++) { gdx[2 * i] = st.du[i] / fLim; gdx[2 * i + 1] = -st.du[i] / fLim; }
      for (let k = 0; k < M2; k++) {
        let a = 0; const row = k * N; for (let j = 0; j < len[k]; j++) a += Ar[row + j] * st.du[j];
        const ds = (rs[k] - Ek[k] * a) / Tk[k], o = nb + 3 * k;
        st.dsig[k] = ds; gdx[o] = a - ds; gdx[o + 1] = -a - ds; gdx[o + 2] = -ds;
      }
      for (let i = 0; i < m; i++) {
        st.dlam[i] = D[i] * (gdx[i] + rp[i]) - rc[i] / sl[i];
        st.ds[i] = -(rc[i] + sl[i] * st.dlam[i]) / lam[i];
      }
    }
    function maxStep(st) {
      let a = 1;
      for (let i = 0; i < m; i++) {
        if (st.ds[i] < 0) a = Math.min(a, -sl[i] / st.ds[i]);
        if (st.dlam[i] < 0) a = Math.min(a, -lam[i] / st.dlam[i]);
      }
      return a;
    }

    function solve(s0) {
      for (let i = 0; i < N; i++) g[i] = F[i * 4] * s0[0] + F[i * 4 + 1] * s0[1] + F[i * 4 + 2] * s0[2] + F[i * 4 + 3] * s0[3];
      for (let k = 0; k < M2; k++) d[k] = Dr[k * 4] * s0[0] + Dr[k * 4 + 1] * s0[1] + Dr[k * 4 + 2] * s0[2] + Dr[k * 4 + 3] * s0[3];
      // Arranque primal factible (u = 0, σ por encima de la violación actual) y dual que anula
      // el residuo de las holguras (λ = w/3 en sus tres filas): con λ = 1, μ explota.
      u.fill(0);
      for (let i = 0; i < nb; i++) { sl[i] = 1; lam[i] = 1; }
      for (let k = 0; k < M2; k++) {
        sig[k] = Math.max(0, Math.abs(d[k]) - 1) + 1;
        const o = nb + 3 * k;
        sl[o] = 1 - d[k] + sig[k]; sl[o + 1] = 1 + d[k] + sig[k]; sl[o + 2] = sig[k];
        lam[o] = lam[o + 1] = lam[o + 2] = wS[k] / 3;
      }
      let iters, converged = false;
      for (iters = 1; iters <= 50; iters++) {
        residuals();
        let mu = 0, pinf = 0, dinf = 0;
        for (let i = 0; i < m; i++) { mu += sl[i] * lam[i]; pinf = Math.max(pinf, Math.abs(rp[i])); }
        mu /= m;
        for (let i = 0; i < N; i++) dinf = Math.max(dinf, Math.abs(rdu[i]));
        for (let k = 0; k < M2; k++) dinf = Math.max(dinf, Math.abs(rds[k]));
        if (opts.debug) console.log(iters, pinf.toExponential(2), dinf.toExponential(2), mu.toExponential(2));
        // Tolerancias relativas a la escala de los multiplicadores (w·lim ≈ 10⁴).
        if (pinf < 1e-9 && dinf < 1e-10 * scaleD && mu < 1e-12 * scaleD) { converged = true; break; }
        // Matriz reducida (σ eliminada por bloques): H + Σ_caja D/F² + Σ_k c_k a_k a_kᵀ.
        for (let i = 0; i < m; i++) D[i] = lam[i] / sl[i];
        Mred.set(H);
        for (let i = 0; i < N; i++) Mred[i * N + i] += (D[2 * i] + D[2 * i + 1]) / (fLim * fLim) + 1e-9;
        for (let k = 0; k < M2; k++) {
          const o = nb + 3 * k, d1 = D[o], d2 = D[o + 1];
          Tk[k] = d1 + d2 + D[o + 2]; Ek[k] = d2 - d1;
          const ck = d1 + d2 - (Ek[k] * Ek[k]) / Tk[k], row = k * N, n = len[k];
          for (let i = 0; i < n; i++) {
            const v = ck * Ar[row + i];
            if (v !== 0) for (let j = 0; j <= i; j++) Mred[i * N + j] += v * Ar[row + j];
          }
        }
        choleskyInPlace(Mred);
        for (let i = 0; i < m; i++) rcA[i] = sl[i] * lam[i];
        newton(rcA, aff);
        const aA = maxStep(aff);
        let muAff = 0;
        for (let i = 0; i < m; i++) muAff += (sl[i] + aA * aff.ds[i]) * (lam[i] + aA * aff.dlam[i]);
        muAff /= m;
        const sigma = Math.min(1, Math.pow(muAff / mu, 3));
        for (let i = 0; i < m; i++) rcC[i] = sl[i] * lam[i] + aff.ds[i] * aff.dlam[i] - sigma * mu;
        newton(rcC, cor);
        const a = Math.min(1, 0.99 * maxStep(cor));
        uSave.set(u); sigSave.set(sig);
        let finite = true;
        for (let i = 0; i < N; i++) { u[i] += a * cor.du[i]; finite = finite && Number.isFinite(u[i]); }
        for (let k = 0; k < M2; k++) sig[k] += a * cor.dsig[k];
        for (let i = 0; i < m; i++) { sl[i] += a * cor.ds[i]; lam[i] += a * cor.dlam[i]; }
        if (!finite) { u.set(uSave); sig.set(sigSave); break; } // nunca devolver NaN: último iterado válido
      }
      // Trayectoria prevista por el modelo lineal, para dibujarla.
      const forces = Array.from(u, (v) => clip(v, -fLim, fLim));
      const plan = [Array.from(s0)];
      let s = Array.from(s0);
      for (let i = 0; i < N; i++) {
        const f = forces[i];
        s = [A[0][0] * s[0] + A[0][1] * s[1] + B[0] * f, A[1][1] * s[1] + A[1][2] * s[2] + B[1] * f,
          A[2][2] * s[2] + A[2][3] * s[3] + B[2] * f, A[3][2] * s[2] + A[3][3] * s[3] + B[3] * f];
        plan.push(s);
      }
      let maxSlack = 0;
      for (let k = 0; k < M2; k++) maxSlack = Math.max(maxSlack, sig[k] * lims[k < N ? 0 : 1]);
      return { force: forces[0], forces, plan, iters, converged, maxSlack };
    }
    return {
      N, P: Pt, solve,
      reset() {},
      act(s) { const sol = solve(s); return { force: clip(sol.force, -P.forceMag, P.forceMag), info: sol }; },
    };
  }

  // ---------------------------------------------------------------- RL tabular (discretization.py, tabular.py)
  const deg = (v) => (v * Math.PI) / 180;
  const BOX_EDGES = [[-0.8, 0.8], [-0.5, 0.5], [deg(-6), deg(-1), 0, deg(1), deg(6)], [deg(-50), deg(50)]];
  const BOX_SHAPE = BOX_EDGES.map((e) => e.length + 1); // 3·3·6·3 = 162
  const FORCE_LEVELS = [-10, -5, 0, 5, 10];
  // Convención de pole.c / searchsorted(side="right"): un valor igual a un borde va a la caja superior.
  function boxIndex(s) {
    let idx = 0;
    for (let v = 0; v < 4; v++) {
      let b = 0;
      while (b < BOX_EDGES[v].length && s[v] >= BOX_EDGES[v][b]) b++;
      idx = idx * BOX_SHAPE[v] + b;
    }
    return idx;
  }
  // Empates hacia la fuerza más suave ("ante la duda, no empujes"), luego izquierda antes que derecha.
  const TIE_ORDER = [2, 1, 3, 0, 4];
  function greedyAction(row) {
    let best = TIE_ORDER[0];
    for (const a of TIE_ORDER) if (row[a] > row[best]) best = a;
    return best;
  }
  function makeTabularPolicy(qTable) {
    return { reset() {}, act(s) { const b = boxIndex(s); const a = greedyAction(qTable[b]); return { force: FORCE_LEVELS[a], info: { box: b, action: a, q: qTable[b] } }; } };
  }

  // Entrenador incremental: Q-learning o SARSA con los hiperparámetros del repo (α = 0.1,
  // γ = 0.99, ε lineal de 1 a 0.01 en la primera mitad). Avanza por episodios para que la
  // interfaz pueda repartir el cálculo entre fotogramas y mostrar cómo aprende.
  function makeTabularTrainer(cfg) {
    const c = Object.assign({ method: "q_learning", episodes: 10000, alpha: 0.1, gamma: 0.99,
      epsStart: 1, epsEnd: 0.01, decayFraction: 0.5, seed: 0, evalEvery: 250, nEval: 10, maxSteps: 500 }, cfg);
    const rand = rng(c.seed * 7919 + 17);
    const Q = Array.from({ length: 162 }, () => new Float64Array(5));
    const visits = new Uint32Array(162);
    let episode = 0;
    const history = { returns: [], evals: [] };
    let best = { mean: -Infinity, episode: 0, q: Q.map((r) => Float64Array.from(r)) };
    const epsilonAt = (ep) => {
      const decay = Math.max(1, Math.round(c.decayFraction * c.episodes));
      return c.epsStart + Math.min(1, ep / decay) * (c.epsEnd - c.epsStart);
    };
    const egreedy = (row, eps) => {
      if (rand() < eps) return Math.floor(rand() * 5);
      let mx = -Infinity; const ties = [];
      for (let a = 0; a < 5; a++) { if (row[a] > mx) { mx = row[a]; ties.length = 0; ties.push(a); } else if (row[a] === mx) ties.push(a); }
      return ties[Math.floor(rand() * ties.length)];
    };
    function evaluateGreedy(table) {
      const pol = makeTabularPolicy(table);
      let total = 0;
      for (let e = 0; e < c.nEval; e++) total += runEpisode(pol, initialState(rng(2000 + e)), c.maxSteps).steps;
      return total / c.nEval;
    }
    // Estado del episodio en curso: permite avanzar paso a paso (para dibujarlo en vivo)
    // o episodio a episodio (entrenamiento rápido) con exactamente la misma actualización TD.
    let cur = null;
    function beginEpisode() {
      const eps = epsilonAt(episode), s0 = initialState(rand), b = boxIndex(s0);
      cur = { eps, s: s0, b, a: egreedy(Q[b], eps), ret: 0, t: 0 };
    }
    function stepOnce() {
      if (cur === null) beginEpisode();
      const e = cur, force = FORCE_LEVELS[e.a];
      e.s = step(e.s, force);
      const term = failure(e.s) !== null;
      const b2 = boxIndex(e.s), a2 = egreedy(Q[b2], e.eps);
      let target = 1; // recompensa de CartPole-v1: +1 por paso
      if (!term) target += c.gamma * (c.method === "q_learning" ? Math.max(...Q[b2]) : Q[b2][a2]);
      Q[e.b][e.a] += c.alpha * (target - Q[e.b][e.a]);
      visits[e.b]++;
      e.ret += 1; e.t++; e.b = b2; e.a = a2;
      const done = term || e.t >= c.maxSteps;
      const out = { s: e.s, force, done, reason: term ? failure(e.s) : null, ret: e.ret };
      if (done) endEpisode();
      return out;
    }
    function endEpisode() {
      episode++;
      history.returns.push(cur.ret);
      cur = null;
      if (episode % c.evalEvery === 0) {
        const mean = evaluateGreedy(Q);
        history.evals.push([episode, mean]);
        if (mean > best.mean) best = { mean, episode, q: Q.map((r) => Float64Array.from(r)) };
      }
    }
    function trainEpisode() {
      let out;
      do out = stepOnce(); while (!out.done);
      return out.ret;
    }
    return { cfg: c, Q, visits, history, get episode() { return episode; }, get best() { return best; },
      epsilon: () => epsilonAt(episode), trainEpisode, stepOnce, done: () => episode >= c.episodes,
      get current() { return cur; } };
  }

  // ---------------------------------------------------------------- redes exportadas (mlp-json-v1)
  // Inferencia documentada en los JSON: x = s / input_scales; capas W·x + b con ReLU salvo la última.
  function mlpForward(net, s) {
    let x = s.map((v, i) => v / net.input_scales[i]);
    net.layers.forEach((layer, li) => {
      const out = layer.W.map((row, i) => row.reduce((acc, wv, j) => acc + wv * x[j], layer.b[i]));
      x = li < net.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
    });
    return x;
  }
  function makeNetPolicy(net) {
    if (net.force_limit !== undefined) { // actor-crítico: F = F_max · tanh(media), sin ruido
      return { reset() {}, act(s) { const o = mlpForward(net, s); return { force: net.force_limit * Math.tanh(o[0]), info: {} }; } };
    }
    return { reset() {}, act(s) { // DQN: acción = nivel de fuerza de mayor Q
      const q = mlpForward(net, s);
      let a = 0; for (let i = 1; i < q.length; i++) if (q[i] > q[a]) a = i;
      return { force: net.force_levels[a], info: { q, action: a } };
    } };
  }

  // ---------------------------------------------------------------- episodio completo (sin interfaz)
  function runEpisode(ctrl, s0, maxSteps = 500) {
    ctrl.reset();
    let s = s0;
    for (let t = 0; t < maxSteps; t++) {
      const f = clip(ctrl.act(s).force, -P.forceMag, P.forceMag);
      s = step(s, f);
      if (failure(s) !== null) return { steps: t + 1, state: s, reason: failure(s) };
    }
    return { steps: maxSteps, state: s, reason: null };
  }

  return { P, step, failure, clip, rng, initialState, THETA_REF_LIMIT, makePID, RULES, LABELS, memberships,
    makeFuzzy, A, B, brysonWeights, dlqr, makeLQR, makeMPC, BOX_EDGES, FORCE_LEVELS, boxIndex, greedyAction,
    makeTabularPolicy, makeTabularTrainer, mlpForward, makeNetPolicy, runEpisode };
})();
if (typeof module !== "undefined") module.exports = Core;
