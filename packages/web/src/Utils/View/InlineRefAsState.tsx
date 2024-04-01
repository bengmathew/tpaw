import { ReactNode, useState } from 'react'

export const InlineRefAsState = <Element,>({
  children,
}: {
  children: (
    x: [ref: Element | null, setRef: (x: Element | null) => void],
  ) => ReactNode
}) => {
  const [ref, setRef] = useState<Element | null>(null)
  return children([ref, setRef])
}
