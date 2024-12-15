import { useEffect, useState } from 'react'
import { WASM, getWASM } from '../../../Simulator/Simulator/GetWASM'
import { asyncEffect } from '../../../Utils/AsyncEffect'

import { ReactNode } from 'react'
import { createContext } from '../../../Utils/CreateContext'

const [Context, useWASM] = createContext<{ wasm: WASM }>('WASM')

export { useWASM }

export const WithWASM = ({ children }: { children: ReactNode }) => {
  const [wasm, setWASM] = useState<WASM | null>(null)
  useEffect(
    () =>
      asyncEffect(async () => {
        const wasm = await getWASM()
        setWASM(wasm)
      }),
    [],
  )
  return (
    wasm && <Context.Provider value={{ wasm }}>{children}</Context.Provider>
  )
}
