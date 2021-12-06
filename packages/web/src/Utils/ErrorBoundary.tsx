import React, { ReactElement, ReactNode } from 'react'

type Props = {
  children: ReactNode
  fallback: ReactElement | ((error: Error) => ReactElement)
}
type State = {error: Error | null}

export class ErrorBoundary extends React.Component<Props, State> {
  state: State = {error: null}

  static getDerivedStateFromError(error: Error): State {
    return {error: error}
  }

  render() {
    const {children, fallback} = this.props
    const {error} = this.state
    return error
      ? typeof fallback === 'function'
        ? fallback(error)
        : fallback
      : children
  }
}
