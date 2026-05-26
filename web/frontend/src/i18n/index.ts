/**
 * @file index.ts
 * @description Internationalization setup for the React frontend application using react-i18next.
 * English is loaded as the default locale, with Turkish supported as a runtime option.
 */

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import enTranslation from './en.json';
import trTranslation from './tr.json';

// Fetch persistent language choice or fallback to English
const savedLanguage = localStorage.getItem('language') || 'en';

i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: {
        translation: enTranslation
      },
      tr: {
        translation: trTranslation
      }
    },
    lng: savedLanguage,
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false // React already escapes string contents
    }
  });

export default i18n;
