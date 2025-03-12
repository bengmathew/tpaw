import { clsx } from 'clsx'
import React from 'react'

export const Scratch = React.memo(({ className }: { className?: string }) => {
  return <_Body />
})

export const _Body = React.memo(({ className }: { className?: string }) => {
  return (
    <div className={clsx(className)}>
      <button className=" btn2-lg btn2-dark rounded-full" onClick={() => {}}>
        run
      </button>
    </div>
  )
})
