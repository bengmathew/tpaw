import React from 'react'
import {Contentful} from '../../Utils/Contentful'
import {AppPage} from '../App/AppPage'
import {Footer} from '../App/Footer'

export const TOS = React.memo(
  ({content, title}: {content: Contentful.FetchedInline; title: string}) => {
    return (
      <AppPage
        className="grid pt-header"
        title={`${title} - TPAW Planner`}
        curr={'other'}
        style={{grid: '1fr auto/auto'}}
      >
        <div className="flex flex-col items-center mb-20 mt-6">
          <div className="w-full max-w-[650px] px-4 z-0">
            <div className=" ">
              <Contentful.RichText
                body={content.fields.body}
                h1="font-bold text-4xl "
                h2="font-bold text-xl mt-6"
                p="mt-4 p-base"
              />
            </div>
          </div>
        </div>
        <Footer />
      </AppPage>
    )
  }
)
