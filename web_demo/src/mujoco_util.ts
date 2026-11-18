import type { MainModule, MjData, MjModel } from "mujoco-js";

export type MujocoModule = MainModule;

/** Copy a MuJoCo embind numeric buffer into a plain Float64Array view. */
export function asArray(buf: any, length: number): Float64Array {
  if (buf instanceof Float64Array) return buf.subarray(0, length);
  if (buf instanceof Float32Array) {
    const out = new Float64Array(length);
    for (let i = 0; i < length; i++) out[i] = buf[i];
    return out;
  }
  // Embind VectorFloat / memory view with length + indexing
  const out = new Float64Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = typeof buf.get === "function" ? Number(buf.get(i)) : Number(buf[i]);
  }
  return out;
}

/** Write a scalar into an embind buffer or TypedArray. */
export function writeNum(buf: any, index: number, value: number): void {
  if (buf instanceof Float64Array || buf instanceof Float32Array) {
    buf[index] = value;
    return;
  }
  if (typeof buf.set === "function" && typeof buf.get === "function") {
    buf.set(index, value);
    return;
  }
  buf[index] = value;
}

export function name2id(
  mj: MujocoModule,
  model: MjModel,
  kind: "body" | "joint" | "actuator" | "sensor" | "camera",
  name: string,
): number {
  const enumMap = {
    body: mj.mjtObj.mjOBJ_BODY,
    joint: mj.mjtObj.mjOBJ_JOINT,
    actuator: mj.mjtObj.mjOBJ_ACTUATOR,
    sensor: mj.mjtObj.mjOBJ_SENSOR,
    camera: mj.mjtObj.mjOBJ_CAMERA,
  } as const;
  const typeVal = enumMap[kind];
  // embind enum may be object with .value
  const typeNum = typeof typeVal === "number" ? typeVal : (typeVal as any).value;
  return mj.mj_name2id(model, typeNum, name);
}

export async function fetchToVfs(
  mj: MujocoModule,
  vfsPath: string,
  url: string,
): Promise<void> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  const dir = vfsPath.slice(0, vfsPath.lastIndexOf("/"));
  mj.FS.mkdirTree(dir, 0o777);
  mj.FS.writeFile(vfsPath, buf);
}

export async function loadPackScene(
  mj: MujocoModule,
  packUrl: string,
): Promise<{ model: MjModel; data: MjData }> {
  // Ensure VFS mount exists (created once in main).
  try {
    mj.FS.mkdir("/working");
  } catch {
    /* exists */
  }
  try {
    mj.FS.mount(mj.MEMFS, { root: "." }, "/working");
  } catch {
    /* already mounted */
  }

  const sceneText = await (await fetch(`${packUrl}/scene.xml`)).text();
  // Collect mesh files referenced as file="..."
  const meshNames = new Set<string>();
  for (const m of sceneText.matchAll(/file="([^"]+\.STL)"/gi)) {
    meshNames.add(m[1]);
  }
  mj.FS.mkdirTree("/working/meshes", 0o777);
  await Promise.all(
    [...meshNames].map((name) =>
      fetchToVfs(mj, `/working/meshes/${name}`, `${packUrl}/meshes/${name}`),
    ),
  );
  mj.FS.writeFile("/working/scene.xml", sceneText);

  // Load from cwd-relative path so meshdir="meshes/" resolves.
  const prev = mj.FS.cwd();
  mj.FS.chdir("/working");
  let model: MjModel;
  try {
    model = mj.MjModel.loadFromXML("scene.xml");
  } finally {
    mj.FS.chdir(prev);
  }
  if (!model) throw new Error("MjModel.loadFromXML returned empty");
  const data = new mj.MjData(model);
  mj.mj_forward(model, data);
  return { model, data };
}
