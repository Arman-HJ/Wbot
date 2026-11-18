/** Quaternion helpers (MuJoCo wxyz). */

export function quatApplyInverse(
  wxyz: ArrayLike<number>,
  vec: ArrayLike<number>,
  out: Float32Array = new Float32Array(3),
): Float32Array {
  // q^{-1} * v * q  for unit quaternion (wxyz)
  const w = wxyz[0];
  const x = -wxyz[1];
  const y = -wxyz[2];
  const z = -wxyz[3];
  const vx = vec[0];
  const vy = vec[1];
  const vz = vec[2];
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  out[0] = vx + w * tx + (y * tz - z * ty);
  out[1] = vy + w * ty + (z * tx - x * tz);
  out[2] = vz + w * tz + (x * ty - y * tx);
  return out;
}

export function quatInv(
  q: ArrayLike<number>,
  out: Float32Array = new Float32Array(4),
): Float32Array {
  out[0] = q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = -q[3];
  return out;
}

export function quatMul(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  out: Float32Array = new Float32Array(4),
): Float32Array {
  const aw = a[0],
    ax = a[1],
    ay = a[2],
    az = a[3];
  const bw = b[0],
    bx = b[1],
    by = b[2],
    bz = b[3];
  out[0] = aw * bw - ax * bx - ay * by - az * bz;
  out[1] = aw * bx + ax * bw + ay * bz - az * by;
  out[2] = aw * by - ax * bz + ay * bw + az * bx;
  out[3] = aw * bz + ax * by - ay * bx + az * bw;
  return out;
}

export function quatApply(
  q: ArrayLike<number>,
  v: ArrayLike<number>,
  out: Float32Array = new Float32Array(3),
): Float32Array {
  return quatApplyInverse(
    [q[0], -q[1], -q[2], -q[3]],
    v,
    out,
  );
}

/** Keep yaw (about +Z) of a wxyz quaternion. */
export function yawQuat(
  q: ArrayLike<number>,
  out: Float32Array = new Float32Array(4),
): Float32Array {
  const w = q[0];
  const z = q[3];
  const n = Math.hypot(w, z) || 1;
  out[0] = w / n;
  out[1] = 0;
  out[2] = 0;
  out[3] = z / n;
  return out;
}

/**
 * Rotation matrix from wxyz, then first two columns flattened row-major
 * (matches mjlab `matrix_from_quat(q)[..., :2].reshape(-1)` → 6-D).
 */
export function matrixFromQuatFirst2Cols(
  q: ArrayLike<number>,
  out: Float32Array = new Float32Array(6),
): Float32Array {
  const w = q[0],
    x = q[1],
    y = q[2],
    z = q[3];
  const r00 = 1 - 2 * (y * y + z * z);
  const r01 = 2 * (x * y - w * z);
  const r10 = 2 * (x * y + w * z);
  const r11 = 1 - 2 * (x * x + z * z);
  const r20 = 2 * (x * z - w * y);
  const r21 = 2 * (y * z + w * x);
  out[0] = r00;
  out[1] = r01;
  out[2] = r10;
  out[3] = r11;
  out[4] = r20;
  out[5] = r21;
  return out;
}

export function cross(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  out: Float32Array = new Float32Array(3),
): Float32Array {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
  return out;
}

/** Angular distance (rad) between two wxyz quaternions — matches quat_error_magnitude. */
export function quatErrorMagnitude(
  q1: ArrayLike<number>,
  q2: ArrayLike<number>,
): number {
  // q_err = inv(q1) * q2, then 2*atan2(|xyz|, w) with shortest path.
  const iw = q1[0],
    ix = -q1[1],
    iy = -q1[2],
    iz = -q1[3];
  const bw = q2[0],
    bx = q2[1],
    by = q2[2],
    bz = q2[3];
  let w = iw * bw - ix * bx - iy * by - iz * bz;
  let x = iw * bx + ix * bw + iy * bz - iz * by;
  let y = iw * by - ix * bz + iy * bw + iz * bx;
  let z = iw * bz + ix * by - iy * bx + iz * bw;
  if (w < 0) {
    w = -w;
    x = -x;
    y = -y;
    z = -z;
  }
  return 2 * Math.atan2(Math.hypot(x, y, z), w);
}
