import { documentToReactComponents } from '@contentful/rich-text-react-renderer'
import { BLOCKS } from '@contentful/rich-text-types'
import { faLongArrowAltRight } from '@fortawesome/pro-regular-svg-icons'
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
}> = async context => ({
  props: {
    detail: await Contentful.fetchInline('6Gv05DQzOKaPAE4QgCqdqh'),
  },
})

export const Index = React.memo(
  ({ detail}: InferGetStaticPropsType<typeof indexGetStaticProps>) => {
    return (
      <div className="font-font1  text-gray-800 px-2 sm:px-20">
        <Head>
          <title>TPAW Planner</title>
        </Head>
        {/* <h2 className="text-2xl font-bold ml-4  mt-2">TPAW Planner</h2> */}
        <div className="flex justify-end sticky top-0 pt-5 ">
          <Link href="/plan">
            <a className="btn-dark btn-lg ">
              Create Your Plan{' '}
              <FontAwesomeIcon className="ml-2" icon={faLongArrowAltRight} />{' '}
            </a>
          </Link>
        </div>
        <div className="  pt-16 ">
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

        <div className="flex flex-col justify-start items-start mt-8">
          <div className="  max-w-[700px]">
            {documentToReactComponents(detail.fields.body, {
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
                  <p className=" text-lg mt-4">{children}</p>
                ),
              },
            })}
          </div>
        </div>
        {/* <div className="flex justify-center my-16">
          <Link href="/plan">
            <a className="btn-dark btn-lg ">
              Create Your Plan{' '}
              <FontAwesomeIcon className="ml-2" icon={faLongArrowAltRight} />{' '}
            </a>
          </Link>
        </div> */}
        {/* <div className="max-w-[700px]">
            <Contentful.RichText
              body={content.fields.body}
              h1="text-4xl font-bold mb-8"
              h2="text-lg font-bold mb-2"
              p="text-lg mb-4"
              a="underline"
            />
            <div className=" mt-10">
              <Link href="/plan">
                <a className="text-2xl px-4 py-2 btn-dark">Create Your Plan</a>
              </Link>
            </div>
          </div> */}

        <Footer className="flex justify-center mt-16  mb-2  gap-x-4 sm:gap-x-4" />
      </div>
    )
  }
)
