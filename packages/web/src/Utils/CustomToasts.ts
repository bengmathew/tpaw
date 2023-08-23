import { ToastOptions, toast } from 'react-toastify'

export const successToast = (message: string) => {
  toast(message, { type: 'success' })
}
export const infoToast = (message: string) => {
  toast(message, { type: 'info' })
}
export const errorToast = (
  message = 'Something went wrong.',
  opts: ToastOptions = {},
) => {
  toast(message, { ...opts, type: 'error' })
}
