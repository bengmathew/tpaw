import { documentToReactComponents } from '@contentful/rich-text-react-renderer'
import { BLOCKS } from '@contentful/rich-text-types'
import { faLongArrowAltRight } from '@fortawesome/pro-solid-svg-icons'
import { faCircle } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { GetStaticProps, InferGetStaticPropsType } from 'next'
import Head from 'next/head'
import Link from 'next/link'
import React from 'react'
import { Contentful } from '../../Utils/Contentful'
import { Footer } from '../App/Footer'

export const indexGetStaticProps: GetStaticProps<{
  detail: Awaited<ReturnType<typeof Contentful.fetchInline>>
}> = async (context) => ({
  props: {
    detail: await Contentful.fetchInline('6Gv05DQzOKaPAE4QgCqdqh'),
  },
})

export const Index = React.memo(
  ({ detail }: InferGetStaticPropsType<typeof indexGetStaticProps>) => {
    return (
      <div className="font-font1  text-gray-800 pl-4 pr-4 sm:pl-20 sm:pr-4">
        <Head>
          <title>TPAW Planner</title>
        </Head>
        <div className="flex justify-end sticky top-0 ">
          <div className="flex items-center gap-x-6 pt-4 pb-2 pl-6 bg-pageBG  rounded-bl-2xl bg-opacity-100 ">
            <Link className=" block text-lg  font-bold" href="/learn">
              Learn
            </Link>
            <Link className="btn-dark btn-lg " href="/plan">
              Create Your Plan{' '}
              <FontAwesomeIcon className="ml-2" icon={faLongArrowAltRight} />{' '}
            </Link>
          </div>
        </div>
        <div className="  pt-10 ">
          <h2 className="text-3xl font-bold">
            <span className="">TPAW Planner</span>{' '}
          </h2>
          <h2 className="">
            <span className="text-base italic ">
              <b>T</b>otal <b>P</b>ortfolio <b>A</b>llocation and <b>W</b>
              ithdrawal
            </span>
          </h2>
        </div>

        <div className="flex flex-col justify-start items-start mt-8 mb-20">
          <div className="  max-w-[700px]">
            {documentToReactComponents(detail.TPAW, {
              renderNode: {
                [BLOCKS.HEADING_2]: (node, children) => (
                  <div className={`flex items-top mt-10`}>
                    <FontAwesomeIcon
                      className="mr-2  text-sm mt-2"
                      icon={faCircle}
                    />
                    <h2 className="font-bold text-lg">{children}</h2>
                  </div>
                ),
                [BLOCKS.PARAGRAPH]: (node, children) => (
                  <p className="p-base mt-4">{children}</p>
                ),
              },
            })}
          </div>
        </div>

        <Footer />
      </div>
    )
  },
)
