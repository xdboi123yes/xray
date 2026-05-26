/**
 * @file pdf-report.ts
 * @description Bilingual high-fidelity A4 clinical diagnostic PDF report exporter.
 * Compiles patient metadata, cascading parameters, and attributions into a print-ready document.
 */

import { jsPDF } from 'jspdf';
import i18n from '../i18n';

interface PDFReportData {
  requestId: string;
  prediction: string;
  confidence: number;
  tierUsed: number;
  mcVariance: number | null;
  timestamp: string;
  originalImageB64?: string;
  gradcamImageB64?: string;
  conformalSet?: string[];
  activeThreshold?: number;
  language: string;
}

export const exportClinicalPdfReport = async (data: PDFReportData): Promise<void> => {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const opt = { lng: data.language };

  // --- Translation Dictionary ---
  const t = {
    title: i18n.t('pdfReport.title', opt),
    subtitle: i18n.t('pdfReport.subtitle', opt),
    uuid: i18n.t('pdfReport.uuid', opt),
    timestamp: i18n.t('pdfReport.timestamp', opt),
    threshold: i18n.t('pdfReport.threshold', opt),
    tier: i18n.t('pdfReport.tier', opt),
    outcome: i18n.t('pdfReport.outcome', opt),
    decision: i18n.t('pdfReport.decision', opt),
    confidence: i18n.t('pdfReport.confidence', opt),
    uncertainty: i18n.t('pdfReport.uncertainty', opt),
    conformal: i18n.t('pdfReport.conformal', opt),
    conformalDesc: i18n.t('pdfReport.conformalDesc', opt),
    imaging: i18n.t('pdfReport.imaging', opt),
    disclaimerTitle: i18n.t('pdfReport.disclaimerTitle', opt),
    disclaimerText: i18n.t('pdfReport.disclaimerText', opt),
    signature: i18n.t('pdfReport.signature', opt),
    pneumoPositive: i18n.t('pdfReport.pneumoPositive', opt),
    normalNegative: i18n.t('pdfReport.normalNegative', opt),
    pneumoLabel: i18n.t('pdfReport.pneumoLabel', opt),
    normalLabel: i18n.t('pdfReport.normalLabel', opt),
    noImage: i18n.t('pdfReport.noImage', opt)
  };

  // --- Document Styling Parameters ---
  const primaryColor = [13, 148, 136]; // Teal #0d9488
  const darkSlate = [15, 23, 42];    // Slate-900
  const lightGrey = [248, 250, 252];  // Slate-50

  // --- Draw Premium Header Banner ---
  doc.setFillColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.rect(0, 0, 210, 38, 'F');

  // Title Text inside Banner
  doc.setTextColor(255, 255, 255);
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(15);
  doc.text(t.title, 14, 18);

  doc.setFont('Helvetica', 'normal');
  doc.setFontSize(9);
  doc.text(t.subtitle, 14, 25);

  // --- Session Metadata Area (A4 Top Right) ---
  doc.setFontSize(7);
  doc.setTextColor(255, 255, 255);
  doc.text(`ThoraxAI v1.0-Production`, 160, 15);
  doc.text(`Defense Release Tag: v1.0`, 160, 20);
  doc.text(`Institutional Clinical DSS`, 160, 25);

  // --- Draw Metadata Details Box ---
  doc.setFillColor(lightGrey[0], lightGrey[1], lightGrey[2]);
  doc.rect(14, 45, 182, 28, 'F');
  doc.setDrawColor(226, 232, 240); // borders
  doc.rect(14, 45, 182, 28, 'S');

  doc.setTextColor(darkSlate[0], darkSlate[1], darkSlate[2]);
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(8);

  // Details text columns
  doc.text(t.uuid, 18, 52);
  doc.setFont('Courier', 'bold');
  doc.setFontSize(8.5);
  doc.text(data.requestId, 55, 52);

  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(8);
  doc.text(t.timestamp, 18, 59);
  doc.setFont('Helvetica', 'normal');
  doc.text(new Date(data.timestamp).toLocaleString(), 55, 59);

  doc.setFont('Helvetica', 'bold');
  doc.text(t.threshold, 18, 66);
  doc.setFont('Helvetica', 'normal');
  doc.text(data.activeThreshold !== undefined ? data.activeThreshold.toFixed(2) : '0.75', 55, 66);

  doc.setFont('Helvetica', 'bold');
  doc.text(t.tier, 115, 66);
  doc.setFont('Helvetica', 'normal');
  doc.text(`Tier ${data.tierUsed} (${data.tierUsed === 1 ? 'MobileNetV2' : 'Ark+ Specialist'})`, 150, 66);

  // --- Diagnostic Outcome Section ---
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(10);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text(t.outcome, 14, 84);

  // Divider Line
  doc.setLineWidth(0.3);
  doc.line(14, 86, 196, 86);

  // Outcomes Box
  doc.setFillColor(lightGrey[0], lightGrey[1], lightGrey[2]);
  doc.rect(14, 90, 182, 22, 'F');
  doc.rect(14, 90, 182, 22, 'S');

  doc.setTextColor(darkSlate[0], darkSlate[1], darkSlate[2]);
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(8.5);
  doc.text(t.decision, 18, 98);

  const isPneumo = data.prediction === 'Pneumothorax';
  doc.setTextColor(isPneumo ? 225 : 16, isPneumo ? 29 : 185, isPneumo ? 72 : 129); // Red for positive, Green for normal
  doc.text(isPneumo ? t.pneumoPositive : t.normalNegative, 60, 98);

  doc.setTextColor(darkSlate[0], darkSlate[1], darkSlate[2]);
  doc.text(t.confidence, 18, 106);
  doc.setFont('Helvetica', 'normal');
  doc.text(`${(data.confidence * 100).toFixed(1)}%`, 60, 106);

  doc.setFont('Helvetica', 'bold');
  doc.text(t.uncertainty, 115, 106);
  doc.setFont('Helvetica', 'normal');
  doc.text(data.mcVariance !== null ? data.mcVariance.toFixed(4) : 'N/A (Bypassed)', 162, 106);

  // --- Conformal Bounds Section ---
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(10);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text(t.conformal, 14, 123);
  doc.line(14, 125, 196, 125);

  doc.setFillColor(lightGrey[0], lightGrey[1], lightGrey[2]);
  doc.rect(14, 129, 182, 20, 'F');
  doc.rect(14, 129, 182, 20, 'S');

  doc.setTextColor(darkSlate[0], darkSlate[1], darkSlate[2]);
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(8.5);
  
  const setLabels = data.conformalSet 
    ? data.conformalSet.map(c => c === 'Pneumothorax' ? t.pneumoLabel : t.normalLabel)
    : [isPneumo ? t.pneumoLabel : t.normalLabel];

  doc.text(`{ ${setLabels.join(' , ')} }`, 20, 137);
  doc.setFont('Helvetica', 'normal');
  doc.setFontSize(7.5);
  doc.setTextColor(100, 116, 139); // Slate-500
  doc.text(t.conformalDesc, 20, 143);

  // --- Imaging / Grad-CAM Saliency Section ---
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(10);
  doc.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
  doc.text(t.imaging, 14, 160);
  doc.line(14, 162, 196, 162);

  // Embed Heatmap Image if provided
  const hasImage = !!(data.originalImageB64 || data.gradcamImageB64);
  if (hasImage) {
    try {
      const imgTarget = data.gradcamImageB64 || data.originalImageB64;
      if (imgTarget) {
        doc.addImage(imgTarget, 'JPEG', 75, 168, 60, 60);
      }
    } catch (err) {
      console.error('Error adding image to PDF:', err);
    }
  } else {
    // Render text placeholder
    doc.setFillColor(240, 240, 240);
    doc.rect(75, 168, 60, 60, 'F');
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text(t.noImage, 90, 198);
  }

  // --- Signature Line ---
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(8.5);
  doc.setTextColor(darkSlate[0], darkSlate[1], darkSlate[2]);
  doc.text(t.signature, 14, 248);
  doc.line(14, 257, 100, 257);

  // --- Footer Disclaimer Panel ---
  doc.setFillColor(lightGrey[0], lightGrey[1], lightGrey[2]);
  doc.rect(14, 263, 182, 22, 'F');
  doc.setDrawColor(230, 230, 230);
  doc.rect(14, 263, 182, 22, 'S');

  doc.setTextColor(180, 83, 9); // Amber-700
  doc.setFont('Helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.text(t.disclaimerTitle, 18, 269);

  doc.setTextColor(100, 116, 139); // Slate-500
  doc.setFont('Helvetica', 'normal');
  doc.setFontSize(6.5);
  
  // Wrap long text
  const splitText = doc.splitTextToSize(t.disclaimerText, 174);
  doc.text(splitText, 18, 273);

  // Save the generated document
  doc.save(`clinical_report_${data.requestId.slice(0, 8)}.pdf`);
};
