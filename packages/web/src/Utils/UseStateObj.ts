import {Dispatch, SetStateAction, useState} from 'react'

export type StateObj<T> = {value: T; set: Dispatch<SetStateAction<T>>}
export function useStateObj<T>(starting: T | (() => T)): StateObj<T> {
  const [value, set] = useState<T>(starting)
  return {value, set}
}
