// 畳み込み(stride 1、バッチ 1)を im2col を作らずに行列積として計算する。
//   out[o, n] = sum_{j,c} w2[o, j*C + c] * act(x[c, n + off(j)]) (+ bias[o]) (+ res[o, n])
//   off(j) = off0 + (j / kw) * rs + (j % kw) * dil
// 1 次元の畳み込みは kw = 近傍の数、rs = 0、off0 = -pad。透かしの 3x3 は縁取り付きの平らな配列(行の幅 rs)上のずらしになる。
// act は Snake(x + inv[c] * sin(alpha[c] * x)^2)か恒等。範囲外の x は 0(act(0) = 0 なので零詰めと同じ)。
// HAS_GATE のときは w2 が conv の行(m_dim 本)のあとに gate の行(m_dim 本)を持ち、
// out[o, n] = (conv + bias[o]) * sigmoid(gate + bias[m_dim + o]) を書く(無効な位置は 0)。
// 1 スレッドの行の前半が conv、後半が同じチャネルの gate になるよう A の行を読み替える。
// 行列積の部分は gemm.wgsl と同じ(A = w2 の vec4 読み、B は x から 4 つずつ集めて共有メモリへ)。
// 各出力は K(j*C + c)の順に足すので、im2col + gemm と同じ結果になる。K は 4 の倍数に詰め物をしてよい(w2 の詰め物の列は 0)。

@group(0) @binding(0) var<storage, read_write> w: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> inv: array<f32>;
@group(0) @binding(5) var<storage, read_write> res: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read_write> dims: array<u32>;

const BM: u32 = {{ bm }}u;
const BNV: u32 = {{ bnv }}u;
const BK: u32 = {{ bk }}u;
const HAS_BIAS: bool = {{ has_bias }};
const HAS_SNAKE: bool = {{ has_snake }};
const HAS_RES: bool = {{ has_res }};
const HAS_GATE: bool = {{ has_gate }};

var<workgroup> As: array<f32, {{ as_len }}>;
var<workgroup> Bs: array<vec4<f32>, {{ bs_len }}>;

fn act(v: f32, c: u32) -> f32 {
    if (HAS_SNAKE) {
        let s = sin(v * alpha[c]);
        return v + inv[c] * (s * s);
    }
    return v;
}

fn load_x(c: u32, t: i32, l_dim: u32) -> f32 {
    if (t >= 0 && u32(t) < l_dim) {
        return act(x[c * l_dim + u32(t)], c);
    }
    return 0.0;
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m_dim = dims[0];      // Cout
    let c_dim = dims[1];      // Cin
    let l_dim = dims[2];      // 入力長
    let n_dim = dims[3];      // 出力長
    let taps = dims[4];
    let dil = i32(dims[5]);
    let off0 = bitcast<i32>(dims[6]);
    let kw = dims[7];
    let rs = i32(dims[8]);
    let k_dim = dims[9];      // taps * Cin を 4 の倍数に切り上げたもの
    let kv = k_dim / 4u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wid.y * select(BM, BM / 2u, HAS_GATE);
    let colv0 = wid.x * BNV;

{{ decl }}

    for (var k0 = 0u; k0 < k_dim; k0 += BK) {
        for (var l = tid; l < BM * (BK / 4u); l += 256u) {
            let r = l / (BK / 4u);
            let kq = l % (BK / 4u);
            var gm = row0 + r;
            var ok = gm < m_dim;
            if (HAS_GATE) {
                let ch = row0 + r % (BM / 2u);
                gm = (r / (BM / 2u)) * m_dim + ch;
                ok = ch < m_dim;
            }
            let gk = k0 + kq * 4u;
            var v = vec4<f32>(0.0);
            if (ok && gk < k_dim) {
                v = w[gm * kv + gk / 4u];
            }
            let base = kq * 4u * BM + r;
            As[base] = v.x;
            As[base + BM] = v.y;
            As[base + 2u * BM] = v.z;
            As[base + 3u * BM] = v.w;
        }
        for (var l = tid; l < BK * BNV; l += 256u) {
            let kr = l / BNV;
            let cv = l % BNV;
            let gk = k0 + kr;
            var v = vec4<f32>(0.0);
            let j = gk / c_dim;
            if (j < taps) {
                let c = gk - j * c_dim;
                let t = i32((colv0 + cv) * 4u) + off0 + i32(j / kw) * rs + i32(j % kw) * dil;
                v = vec4<f32>(load_x(c, t, l_dim), load_x(c, t + 1, l_dim), load_x(c, t + 2, l_dim), load_x(c, t + 3, l_dim));
            }
            Bs[kr * BNV + cv] = v;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < BK; kk++) {
            let ao = kk * BM + ty;
            let bo = kk * BNV + tx;
{{ body }}
        }
        workgroupBarrier();
    }

{{ store }}
}

// 透かしの平らな配列で、出力の位置 n が有効か(行内の列 < w かつ全体の列数 < ncols)
fn valid(n: u32) -> bool {
    return n < dims[10] && n % dims[11] < dims[12];
}

fn store_gate(o: u32, n0: u32, v: vec4<f32>, g: vec4<f32>, m_dim: u32, n_dim: u32) {
    if (o >= m_dim) {
        return;
    }
    let bv = bias[o];
    let bg = bias[m_dim + o];
    for (var e = 0u; e < 4u; e++) {
        let n = n0 + e;
        if (n < n_dim) {
            var y = 0.0;
            if (valid(n)) {
                y = (v[e] + bv) / (1.0 + exp(-(g[e] + bg)));
            }
            // 使わない残差もシェーダから参照しておく(参照がないと束縛が消えて、起動時の束縛の数と合わなくなる)
            if (HAS_RES) {
                y += res[o * n_dim + n];
            }
            out[o * n_dim + n] = y;
        }
    }
}

fn store(o: u32, n0: u32, v: vec4<f32>, m_dim: u32, n_dim: u32) {
    if (o >= m_dim) {
        return;
    }
    var b = 0.0;
    if (HAS_BIAS) {
        b = bias[o];
    }
    for (var e = 0u; e < 4u; e++) {
        let n = n0 + e;
        if (n < n_dim) {
            var y = v[e] + b;
            if (HAS_RES) {
                y += res[o * n_dim + n];
            }
            out[o * n_dim + n] = y;
        }
    }
}
