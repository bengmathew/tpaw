export const yourOrYourPartners = (
  person: 'person1' | 'person2' | { person: 'person1' | 'person2' },
): string =>
  typeof person === 'string'
    ? person === 'person1'
      ? 'your'
      : `your partner's`
    : yourOrYourPartners(person.person)
