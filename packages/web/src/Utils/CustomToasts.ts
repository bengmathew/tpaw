import { toast } from 'react-toastify'

export const infoToast = (message: string) => {
  toast(message, { type: 'info' })
}
export const errorToast = (message = 'Something went wrong.') => {
  toast(message, { type: 'error' })
}
