// Thanks: https://maketintsandshades.com/about#calculation-method
export const makeShades = (base: string) => {
  const rgb = RGB.fromHex(base)
  const _tint = (t: number) => tint(t, rgb)
  const _shade = (t: number) => shade(t, rgb)
  return [
    _tint(1),
    _tint(0.9),
    _tint(0.8),
    _tint(0.7),
    _tint(0.6),
    _tint(0.5),
    _tint(0.4),
    _tint(0.3),
    _tint(0.2),
    _tint(0.1),
    RGB.fromHex(base), // index 10
    _shade(0.9),
    _shade(0.8),
    _shade(0.7),
    _shade(0.6),
    _shade(0.5),
    _shade(0.4),
    _shade(0.3),
    _shade(0.2),
    _shade(0.1),
    _shade(0), // index 20
  ].map(RGB.toHex)
}

export const tint = <R extends RGB | RGBA>(t: number, rgb: R): R => {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
  return RGB.map(rgb as any, (c: number) =>
    Math.round(c + (255 - c) * t),
  ) as unknown as R
}
export const shade = <R extends RGB | RGBA>(s: number, rgb: R): R => {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
  return RGB.map(rgb as any, (c: number) => Math.round(c * s)) as unknown as R
}

export type RGB = readonly [r: number, g: number, b: number]
export type RGBA = readonly [r: number, g: number, b: number, a: number]

export namespace RGB {
  export function map<T>(rgb: RGBA, fn: (x: number) => T): [T, T, T, number]
  export function map<T>(rgb: RGB, fn: (x: number) => T): [T, T, T]
  export function map<T>(
    rgb: RGB | RGBA,
    fn: (x: number) => T,
  ): [T, T, T] | [T, T, T, number] {
    if (rgb.length === 3) {
      return [fn(rgb[0]), fn(rgb[1]), fn(rgb[2])] as [T, T, T]
    } else {
      const [r, g, b, a] = rgb
      return [...map([r, g, b], fn), a] as [T, T, T, number]
    }
  }

  export const addAlpha = ([r, g, b]: RGB, a: number) => [r, g, b, a] as RGBA

  export const toHex = (x: RGB | RGBA) => {
    const rgbHex = `#${x
      .slice(0, 3)
      .map((x: number) => x.toString(16).padStart(2, '0').slice(0, 2))
      .join('')}`

    const alphaToHex = (x: number) =>
      x === 0
        ? '00'
        : x === 1
        ? 'aa'
        : (x * 256).toString(16).padStart(2, '0').slice(0, 2)
    return x.length === 3 ? rgbHex : `${rgbHex}${alphaToHex(x[3])}`
  }

  export const fromHex = (hex: string) =>
    RGB.map([1, 3, 5], (i: number) => parseInt(hex.substring(i, i + 2), 16))

  export const addHex = (rgb: RGB) => ({ rgb, hex: toHex(rgb) })
}
