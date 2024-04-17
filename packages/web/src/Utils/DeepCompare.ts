export function deepCompare(obj1: any, obj2: any, tolerance: number) {
  if (obj1 === obj2) {
    return true
  }

  if (typeof obj1 === 'number' && typeof obj2 === 'number') {
    return Math.abs(obj1 - obj2) < tolerance
  }

  if (
    typeof obj1 !== 'object' ||
    obj1 === null ||
    typeof obj2 !== 'object' ||
    obj2 === null
  ) {
    return false
  }

  if (Array.isArray(obj1) && Array.isArray(obj2)) {
    if (obj1.length !== obj2.length) {
      return false
    }

    for (let i = 0; i < obj1.length; i++) {
      if (!deepCompare(obj1[i], obj2[i], tolerance)) {
        return false
      }
    }
    return true
  } else {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
    const keys1 = Object.keys(obj1)
    // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
    const keys2 = Object.keys(obj2)

    if (keys1.length !== keys2.length) {
      return false
    }

    for (const key of keys1) {
      if (!keys2.includes(key)) {
        return false
      }
      if (!deepCompare(obj1[key], obj2[key], tolerance)) return false
    }
    return true
  }
}
