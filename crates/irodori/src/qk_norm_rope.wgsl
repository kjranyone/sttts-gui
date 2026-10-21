// 注意の q / k: ヘッドごとの RMSNorm(重み付き)と RoPE(前半 rot_heads 個のヘッドだけ)を 1 回で行う。
// 入力 y は [rows, in_stride] の offset 列目から heads*hd 列(q, k, v, gate をまとめた射影の一部でもよい)。
// 出力は [rows, heads*hd]([B, S, H, Dh] と同じ並び)。RoPE は (x_{2i}, x_{2i+1}) を
//   (a*cos - c*sin, a*sin + c*cos) に回す(dit.rs の Rope::apply と同じ式・同じ順)。

@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> cosv: array<f32>;
@group(0) @binding(3) var<storage, read_write> sinv: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> dims: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let rows = dims[0];
    let seq = dims[1];
    let heads = dims[2];
    let hd = dims[3];
    let rot_heads = dims[4];
    let in_stride = dims[5];
    let offset = dims[6];
    let eps = bitcast<f32>(dims[7]);
    let idx = gid.x + gid.y * nwg.x * 64u;
    if (idx >= rows * heads) {
        return;
    }
    let row = idx / heads;
    let h = idx % heads;
    let s = row % seq;
    let src = row * in_stride + offset + h * hd;
    let dst = (row * heads + h) * hd;
    let wb = h * hd;
    var ss = 0.0;
    for (var i = 0u; i < hd; i++) {
        let v = y[src + i];
        ss += v * v;
    }
    let inv = 1.0 / sqrt(ss / f32(hd) + eps);
    if (h < rot_heads) {
        let half = hd / 2u;
        for (var i = 0u; i < half; i++) {
            let a = y[src + 2u * i] * inv * w[wb + 2u * i];
            let c = y[src + 2u * i + 1u] * inv * w[wb + 2u * i + 1u];
            let cs = cosv[s * half + i];
            let sn = sinv[s * half + i];
            out[dst + 2u * i] = a * cs - c * sn;
            out[dst + 2u * i + 1u] = a * sn + c * cs;
        }
    } else {
        for (var i = 0u; i < hd; i++) {
            out[dst + i] = y[src + i] * inv * w[wb + i];
        }
    }
}
