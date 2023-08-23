// Thanks: https://maketintsandshades.com/about#calculation-method
export const makeShades = (base: string) => {
  const rgb = _hexToRgb(base)
  const tint = (t: number) =>
    _mapRGB(rgb, (c: number) => Math.round(c + (255 - c) * t))
  const shade = (s: number) => _mapRGB(rgb, (c: number) => Math.round(c * s))
  return [
    tint(1),
    tint(0.9),
    tint(0.8),
    tint(0.7),
    tint(0.6),
    tint(0.5),
    tint(0.4),
    tint(0.3),
    tint(0.2),
    tint(0.1),
    _hexToRgb(base), // index 10
    shade(0.9),
    shade(0.8),
    shade(0.7),
    shade(0.6),
    shade(0.5),
    shade(0.4),
    shade(0.3),
    shade(0.2),
    shade(0.1),
    shade(0), // index 20
  ].map(_rgbToHex)
}

type _RGB = readonly [r: number, g: number, b: number]

const _mapRGB = <T>([r, g, b]: _RGB, fn: (x: number) => T) =>
  [fn(r), fn(g), fn(b)] as const

const _hexToRgb = (hex: string) =>
  _mapRGB([1, 3, 5], (i: number) => parseInt(hex.substring(i, i + 2), 16))

const _rgbToHex = (rgb: _RGB) =>
  `#${_mapRGB(rgb, (x: number) => x.toString(16).padStart(2, '0')).join('')}`
