import React, { ReactElement, useEffect, useState } from 'react'
import { createContext } from './CreateContext'

const [Context, useWindowWidth] = createContext<number>('WindowSize')

export { useWindowWidth }

export const WithWindowWidth = React.memo(
  ({children}: {children: ReactElement}) => {
    const [value, setValue] = useState(() => window.innerWidth)

    useEffect(() => {
      const handleResize = () => setValue(window.innerWidth)
      window.addEventListener('resize', handleResize)
      return () => window.removeEventListener('resize', handleResize)
    }, [])

    return <Context.Provider value={value}>{children}</Context.Provider>
  }
)
