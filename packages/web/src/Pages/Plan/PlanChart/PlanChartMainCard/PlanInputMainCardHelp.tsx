import Link from 'next/link'
import React from 'react'
import { useGetSectionURL } from '../../Plan'

export const PlanInputMainCardHelp = React.memo(() => {
  const getSectionURL = useGetSectionURL()
  return (
    <Link className="" href={getSectionURL('help')} shallow>
      Help me understand these results.
    </Link>
  )
})
