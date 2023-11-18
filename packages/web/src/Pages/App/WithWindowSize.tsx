import React, { ReactElement, useEffect, useState } from 'react'
import { createContext } from '../../Utils/CreateContext'

const [Context, useContext] = createContext<{
  width: number
  height: number
}>('WindowSize')

export const useWindowSize = () => {
  const windowSize = useContext()
  // Same as tailwind breakpoints https://tailwindcss.com/docs/screens.
  const windowWidthName =
    windowSize.width < 640
      ? ('xs' as const)
      : windowSize.width < 768
      ? ('md' as const)
      : windowSize.width < 1024
      ? ('lg' as const)
      : windowSize.width < 1280
      ? ('xl' as const)
      : windowSize.width < 1536
      ? ('2xl' as const)
      : ('3xl' as const)
  return { windowSize, windowWidthName }
}

export const WithWindowSize = React.memo(
  ({ children }: { children: ReactElement }) => {
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
  },
)
WithWindowSize.displayName = 'WithWindowSize'

const _get = () => ({ width: window.innerWidth, height: window.innerHeight })
