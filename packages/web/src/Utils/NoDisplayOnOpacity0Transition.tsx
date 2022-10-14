import React, {CSSProperties, ReactNode, useEffect, useState} from 'react'

type CSSPropertiesWithOpacity = Exclude<CSSProperties, 'opacity'> & {
  opacity: string
}

type _State = {hidden: boolean; style: CSSPropertiesWithOpacity}
export const NoDisplayOnOpacity0Transition = React.memo(
  ({
    className = '',
    style,
    children,
    onHidden,
    onTransitionEnd,
    noDisplayMeans = 'return <></>',
  }: {
    className?: string
    style: CSSPropertiesWithOpacity
    children: ReactNode
    onTransitionEnd?: () => void
    onHidden?: () => void
    noDisplayMeans?: 'visibility:hidden' | 'return <></>'
  }) => {
    const [state, setState] = useState<_State>(
      style.opacity === '0' ? {hidden: true, style} : {hidden: false, style}
    )
    useEffect(() => {
      setState(prev => {
        if (!prev.hidden) return {hidden: false, style}
        if (style.opacity === '0') return {hidden: true, style}
        window.setTimeout(() => setState({hidden: false, style}), 5)
        return {hidden: false, style: prev.style, targetStyle: style}
      })
    }, [style])

    if (state.hidden && noDisplayMeans === 'return <></>') return <></>
    return (
      <div
        className={`${className}`}
        style={{
          ...state.style,
          visibility:
            state.hidden && noDisplayMeans === 'visibility:hidden'
              ? 'hidden'
              : undefined,
        }}
        onTransitionEnd={() => {
          if (state.style.opacity === '0') {
            setState({hidden: true, style: state.style})
            onHidden?.()
          }
          onTransitionEnd?.()
        }}
      >
        {children}
      </div>
    )
  }
)