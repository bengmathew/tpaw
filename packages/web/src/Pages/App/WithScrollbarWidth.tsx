import React, { ReactElement, useLayoutEffect, useState } from 'react'
import { createContext } from '../../Utils/CreateContext'

const [Context, useScrollbarWidth] = createContext<number>('ScrollbarWidth')

export { useScrollbarWidth }

export const WithScrollbarWidth = React.memo(
  ({ children }: { children: ReactElement }) => {
    const [value, setValue] = useState(0)

    useLayoutEffect(() => {
      // Thanks https://stackoverflow.com/a/986977/2771609.
      const inner = document.createElement('p')
      inner.style.width = '100%'
      inner.style.height = '200px'

      const outer = document.createElement('div')
      outer.style.position = 'absolute'
      outer.style.top = '0px'
      outer.style.left = '0px'
      outer.style.visibility = 'hidden'
      outer.style.width = '200px'
      outer.style.height = '150px'
      outer.style.overflow = 'hidden'
      outer.appendChild(inner)

      document.body.appendChild(outer)
      const w1 = inner.offsetWidth
      outer.style.overflow = 'scroll'
      let w2 = inner.offsetWidth
      if (w1 == w2) w2 = outer.clientWidth

      document.body.removeChild(outer)

      setValue(w1 - w2)
    }, [])

    return <Context.Provider value={value}>{children}</Context.Provider>
  },
)
WithScrollbarWidth.displayName = 'WithScrollbarWidth'
