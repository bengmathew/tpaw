import {useState} from 'react'

export function useIsMouseInside() {
  const [isInside, setIsInside] = useState(false)
  const props = {
    onMouseEnter: (e: React.MouseEvent) => setIsInside(count => true),
    onMouseLeave: (e: React.MouseEvent) => setIsInside(count => false),
  }
  return {isInside, props}
}
