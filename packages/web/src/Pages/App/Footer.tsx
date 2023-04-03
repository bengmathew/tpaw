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
        <Link className="" href="/about">
          About
        </Link>
        <Link className="" href="/license">
          License
        </Link>
        <Link className="" href="/disclaimer">
          Disclaimer
        </Link>
        <Link className="" href="/privacy">
          Privacy
        </Link>
      </div>
    )
  },
)
