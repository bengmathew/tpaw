import * as React from 'react'
import { nundef } from './Utils'

export function createContext<T>(prefix: string) {
  const context = React.createContext<T | undefined>(undefined)
  context.displayName =
    prefix + (prefix.indexOf('Context') === -1 ? 'Context' : '')
  const useContext = () => nundef(React.useContext(context))
  return [context, useContext] as const
}
