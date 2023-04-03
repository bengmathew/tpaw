import React, { ReactElement, useEffect, useState } from 'react'
import { WASM, getWASM } from '../../TPAWSimulator/Worker/GetWASM'
import { asyncEffect } from '../../Utils/AsyncEffect'
import { createContext } from '../../Utils/CreateContext'

const [Context, useWASM] = createContext<WASM>('WASM')

export { useWASM }

export const WithWASM = React.memo(
  ({ children }: { children: ReactElement }) => {
    const [value, setValue] = useState<WASM | null>(null)

    useEffect(() => {
      return asyncEffect(async () => {
        const wasm = await getWASM()
        setValue(wasm)
      })
    }, [])

    if (value === null) return <></>
    return <Context.Provider value={value}>{children}</Context.Provider>
  },
)
