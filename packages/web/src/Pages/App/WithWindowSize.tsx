import React, {ReactElement, useEffect, useState} from 'react'
import {createContext} from '../../Utils/CreateContext'

const [Context, useWindowSize] = createContext<{width: number; height: number}>(
  'WindowSize'
)

export {useWindowSize}

export const WithWindowSize = React.memo(
  ({children}: {children: ReactElement}) => {
    const [value, setValue] = useState(_get)

    useEffect(() => {
      const handleResize = () => setValue(_get())
      window.addEventListener('resize', handleResize)
      window.addEventListener('orientationchange', handleResize)
      return () => {
        window.removeEventListener('resize', handleResize)
        window.removeEventListener('orientationchange', handleResize)
      }
    }, [])

    return <Context.Provider value={value}>{children}</Context.Provider>
  }
)

const _get = () => ({width: window.innerWidth, height: window.innerHeight})
