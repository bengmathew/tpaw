import React, { ReactElement, useEffect, useState } from 'react'
import { createContext } from '../../Utils/CreateContext'
import { Size } from '../../Utils/Geometry'

type SystemInfo = {
  windowSize: Size
  windowWidthName: 'xs' | 'md' | 'lg' | 'xl' | '2xl' | '3xl'
  isPrinting: boolean
}

const [Context, useSystemInfo] = createContext<SystemInfo>('SystemInfo')
export { useSystemInfo }

export const WithSystemInfo = React.memo(
  ({ children }: { children: ReactElement }) => {
    const [value, setValue] = useState<SystemInfo>({
      ..._getWindowSize(),
      isPrinting: false,
    })

    useEffect(() => {
      const handleBeforePrint = () =>
        setValue((prev) => ({ ...prev, isPrinting: true }))
      const handleAfterPrint = () =>
        setValue((prev) => ({ ...prev, isPrinting: false }))
      window.addEventListener('beforeprint', handleBeforePrint)
      window.addEventListener('afterprint', handleAfterPrint)
      return () => {
        window.removeEventListener('beforeprint', handleBeforePrint)
        window.removeEventListener('afterprint', handleAfterPrint)
      }
    }, [])

    useEffect(() => {
      const handleResize = () =>
        setValue((prev) => ({ ...prev, ..._getWindowSize() }))
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
WithSystemInfo.displayName = 'WithWindowSize'

const _getWindowSize = () => {
  const windowSize = {
    width: window.innerWidth,
    height: window.innerHeight,
  }
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
