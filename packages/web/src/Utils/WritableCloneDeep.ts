import _ from 'lodash'

type DeepWriteable<T> = { -readonly [P in keyof T]: DeepWriteable<T[P]> }

export const writableCloneDeep = <T>(x: T): DeepWriteable<T> => {
  return _.cloneDeep(x)
}
