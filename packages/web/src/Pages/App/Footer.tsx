import Link from 'next/link'
import React from 'react'

export const Footer = React.memo(
  ({
    className = 'flex text-sm lighten justify-center py-2  gap-x-4 ',
  }: {
    className?: string
  }) => {
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
  }
)
