import Link from 'next/link'
import React from 'react'
import {useGetSectionURL} from '../../Plan'

export const PlanInputMainCardHelp = React.memo(() => {
  const getSectionURL = useGetSectionURL()
  return (
    <Link href={getSectionURL('results')} shallow>
      <a className="">Help me understand these results.</a>
    </Link>
  )
})
