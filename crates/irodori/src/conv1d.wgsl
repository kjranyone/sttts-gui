// 1 次元畳み込み(stride 1、バッチ 1)を im2col を作らずに行列積として計算する。
//   out[o, n] = sum_{j,c} w2[o, j*C + c] * act(x[c, n + j*dil - pad]) (+ bias[o]) (+ res[o, n])
// act は Snake(x + inv[c] * sin(alpha[c] * x)^2)か恒等。範囲外の x は 0(act(0) = 0 なので零詰めと同じ)。
// 行列積の部分は gemm.wgsl と同じ(A = w2 の vec4 読み、B は x から 4 つずつ集めて共有メモリへ)。
// 各出力は K(j*C + c)の順に足すので、im2col + gemm と同じ結果になる。

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
    let k_dim = dims[4] * c_dim;
    let dil = i32(dims[5]);
    let pad = i32(dims[6]);
    let kv = k_dim / 4u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let row0 = wid.y * BM;
    let colv0 = wid.x * BNV;

{{ decl }}

    for (var k0 = 0u; k0 < k_dim; k0 += BK) {
        for (var l = tid; l < BM * (BK / 4u); l += 256u) {
            let r = l / (BK / 4u);
            let kq = l % (BK / 4u);
            let gm = row0 + r;
            let gk = k0 + kq * 4u;
            var v = vec4<f32>(0.0);
            if (gm < m_dim && gk < k_dim) {
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
            if (gk < k_dim) {
                let j = gk / c_dim;
                let c = gk - j * c_dim;
                let t = i32((colv0 + cv) * 4u) + i32(j) * dil - pad;
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
