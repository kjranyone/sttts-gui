// 転置畳み込み(kernel = 2 * stride)の重なり加算。z は [k*Cout, L](行 = j*Cout + o、j はカーネル内の位置)。
//   out[o, p] = bias[o] + z[r*Cout + o, t] + z[(r+s)*Cout + o, t-1]   (q = p + pad, t = q / s, r = q % s)
// 範囲外の t の項は 0。

@group(0) @binding(0) var<storage, read_write> z: array<f32>;
@group(0) @binding(1) var<storage, read_write> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read_write> dims: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let co = dims[0];
    let l = dims[1];
    let lout = dims[2];
    let s = dims[3];
    let pad = dims[4];
    let p = gid.x;
    let o = gid.y + gid.z * 65535u;
    if (p >= lout || o >= co) {
        return;
    }
    let q = p + pad;
    let t = q / s;
    let r = q % s;
    var v = bias[o];
    if (t < l) {
        v += z[(r * co + o) * l + t];
    }
    if (t >= 1u && t - 1u < l) {
        v += z[((r + s) * co + o) * l + t - 1u];
    }
    out[o * lout + p] = v;
}
