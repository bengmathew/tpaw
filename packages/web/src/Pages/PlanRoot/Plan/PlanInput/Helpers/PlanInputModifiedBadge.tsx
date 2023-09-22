import React from 'react'

export const PlanInputModifiedBadge = React.memo(
  ({ show, mainPage }: { show: boolean; mainPage: boolean }) => {
    if (!show) return <></>
    return (
      <h2 className={`absolute right-2 text-sm ${mainPage?'top-1.5':'-top-1.5 sm:top-1.5'}  bg-orange-300 px-2 rounded-lg `}>
        modified
      </h2>
    )
  },
)
