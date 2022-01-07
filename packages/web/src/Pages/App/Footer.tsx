import Link from 'next/link'
import React from 'react'

export const Footer = React.memo(({className = ''}: {className?: string}) => {
  return (
    <div className={`${className}`}>
      <Link href="/about">
        <a className="">About</a>
      </Link>
      <Link href="/license">
        <a className="">License</a>
      </Link>
      <Link href="/disclaimer">
        <a className="">Disclaimer</a>
      </Link>
      <Link href="/privacy">
        <a className="">Privacy</a>
      </Link>
    </div>
  )
})
