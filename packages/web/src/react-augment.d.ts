import React from "react"

// Thanks: https://stackoverflow.com/a/58473012.
declare module "react" {
  // eslint-disable-next-line 
  function forwardRef<T, P = {}>(
    render: (props: P, ref: ForwardedRef<T>) => ReactElement | null
  ): (props: P & RefAttributes<T>) => ReactElement | null
}