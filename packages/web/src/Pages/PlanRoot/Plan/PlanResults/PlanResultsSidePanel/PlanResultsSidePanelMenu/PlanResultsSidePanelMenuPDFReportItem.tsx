import { faFilePdf } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import React, { useMemo } from 'react'

export const PlanResultsSidePanelMenuPDFReportItem = React.memo(() => {
  const path = useRouter().asPath
  const printURL = useMemo(() => {
    const result = new URL(path, window.location.origin)
    result.searchParams.set('pdf-report', 'true')
    return result
  }, [path])

  return (
    <Menu.Item>
      <Link className="context-menu-item" shallow href={printURL}>
        <span className="inline-block w-[30px]">
          <FontAwesomeIcon icon={faFilePdf} />
        </span>
        PDF Report
      </Link>
    </Menu.Item>
  )
})
