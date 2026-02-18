#!/usr/bin/env python3
"""Generate VeriCall whitepaper PDF - single page."""

from fpdf import FPDF, XPos, YPos

class WhitepaperPDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, 'VeriCall  |  Confidential  |  2025', align='C')

pdf = WhitepaperPDF('P', 'mm', 'Letter')
pdf.set_auto_page_break(auto=False)
pdf.add_page()
pdf.set_margins(22, 15, 22)
pdf.set_x(22)

LH = 4.3  # line height for body text
SPC = 2   # spacing after sections

def heading(text):
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

def body(text):
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, LH, text)

def layer(name, desc):
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(6)
    pdf.cell(0, LH, name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_x(28)
    pdf.multi_cell(pdf.w - 28 - 22, LH, desc)
    pdf.ln(0.5)

# Title
pdf.set_font('Helvetica', 'B', 24)
pdf.set_text_color(20, 60, 120)
pdf.cell(0, 12, 'VeriCall', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

pdf.set_font('Helvetica', '', 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 6, 'Real-Time Voice Authentication for the Age of AI',
         new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
pdf.ln(2)

# Divider
pdf.set_draw_color(20, 60, 120)
pdf.set_line_width(0.5)
pdf.line(22, pdf.get_y(), pdf.w - 22, pdf.get_y())
pdf.ln(4)

# The Problem
heading('The Problem')
body('Voice cloning technology has advanced to the point where a synthetic copy of any voice '
     'can be generated from just seconds of sample audio. Fraudsters are already using cloned '
     'voices to impersonate executives, family members, and public figures over phone calls, '
     'bypassing the innate trust we place in a familiar voice. Existing caller ID and network-level '
     'protections offer no defense against a convincing clone speaking in real time.')
pdf.ln(SPC)

# How VeriCall Works
heading('How VeriCall Works')
body('VeriCall is a mobile application that authenticates callers continuously and in real time. '
     'When both parties have VeriCall installed, the app establishes a cryptographically verified '
     'peer-to-peer audio channel alongside the standard phone call. Every few seconds of live '
     'speech is analyzed on-device and the caller\'s identity is confirmed or flagged, with no '
     'human effort required beyond placing the call.')
pdf.ln(1)
body('Each user enrolls by providing a short voice sample during onboarding. VeriCall extracts '
     'a compact voiceprint, a mathematical signature unique to that individual\'s voice, much '
     'like a fingerprint is unique to a finger. Just as no two fingerprints share the same '
     'pattern of ridges, no two voiceprints share the same acoustic geometry. This voiceprint '
     'is stored securely on the device and used as the reference stamp against which all '
     'future calls are measured.')
pdf.ln(SPC)

# Voiceprint Technology
heading('Voiceprint Technology: Three Layers of Defense')
body('VeriCall\'s voice engine uses a layered architecture that combines speed with depth:')
pdf.ln(1)

layer('Layer 1: Acoustic Screening',
      'A fast rule-based check on spectral and temporal properties of audio. Synthetic speech '
      'leaves subtle artifacts like unnatural smoothness and missing micro-variations that '
      'this layer catches in milliseconds.')

layer('Layer 2: Clone Detection',
      'A machine learning classifier trained extensively on phone-quality audio spanning '
      'dozens of modern attack types, including AI-generated clones, text-to-speech, and '
      'voice conversion. Achieves over 97% detection accuracy on real-world phone-codec audio.')

layer('Layer 3: Speaker Identity',
      'A neural network computes a high-dimensional voiceprint embedding from live speech and '
      'compares it against the enrolled reference, confirming not just that the voice is real, '
      'but that it belongs to the expected speaker.')
pdf.ln(SPC)

# Solving Latency
heading('Dual-Path Audio: RTC for You, MoQ for the AI')
body('VeriCall splits the audio stream into two parallel paths. The caller hears the conversation '
     'over standard WebRTC, the same low-latency protocol used by every major calling app. '
     'Simultaneously, a second copy of the audio is routed through MoQ (Media over QUIC), a '
     'next-generation transport built on the same foundation as HTTP/3. This MoQ stream delivers '
     'higher-fidelity audio with sub-50ms latency directly to the voiceprint engine, giving the '
     'AI the cleanest possible signal for authentication without touching the call audio the user '
     'hears. Combined with on-device neural inference, the full verification loop from microphone '
     'to verdict completes in under one second, entirely in the background.')
pdf.ln(SPC)

# Security Model
heading('Security Model')
body('Every VeriCall device generates a unique cryptographic key pair during setup. Calls are '
     'authenticated at both the device and voice level: the device proves its identity through '
     'digital signatures, and the voice proves the speaker\'s identity through voiceprint matching. '
     'Voiceprints never leave the device. Audio analysis runs entirely on-device, and only the '
     'verdict is shared. Even a compromised network cannot forge a caller\'s identity.')
pdf.ln(SPC)

# Licensing
heading('Licensing & Integration')
body('VeriCall\'s voiceprint engine is designed as a modular SDK suitable for integration into '
     'communication platforms, banking apps, call centers, and enterprise security systems. '
     'The technology is optimized for mobile GPU acceleration on both iOS and Android.')
pdf.ln(4)

# Contact
pdf.set_font('Helvetica', 'I', 8)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 4, 'For licensing inquiries, contact the VeriCall team.', align='C',
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Save
out_path = '/Users/reeceway/Desktop/VeriCall_Whitepaper.pdf'
pdf.output(out_path)
print(f'Saved to {out_path}')
print(f'Pages: {pdf.pages_count}')
