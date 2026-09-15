"""Element-wise justification, v0: which ingredients of the offence do the facts show?

A section is not one thing. IPC 379 (theft) requires dishonest intention,
moveable property, taking it out of someone's possession, and no consent. The
research core's distinctive output (PLAN P3, ARCHITECTURE.md §5
`section_candidate.elements`) is not "379 applies" but "379: dishonest intention
YES ('stole'), moveable property YES ('mobile phone'), without consent UNCLEAR".
That is what lets an officer -- or a court -- see *why*, and see what is missing.

This module is the **rule-based v0** of that checker. Each covered section has a
hand-authored element list; each element has lexical cues in English and Tamil.
Two rules keep it honest:

1. **Never say "no".** A missing cue is an absence of evidence, not evidence of
   absence -- the complainant may simply not have used the word. Absence yields
   `unclear`, and `unclear` is the officer's cue to ask.

2. **Every "yes" carries provenance.** The justification quotes the matched
   words and their offsets, so the claim can be checked against the transcript.

Coverage is the common FIR offences only. Sections without an element list get
`elements=[]`, which the form renders as "elements not analysed" -- the LLM
verifier that replaces this will cover the rest, against the same schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fir.schema.if1 import ElementCheck, Provenance


@dataclass(frozen=True, slots=True)
class Element:
    name: str
    cues: tuple[str, ...]          # regex fragments, case-insensitive, EN + TA
    note: str = ""                 # what the officer should establish if unclear


# IPC section -> ordered elements. Kept to the ingredients a statute actually
# requires, phrased the way a charge sheet would.
ELEMENTS: dict[str, list[Element]] = {
    "379": [  # theft
        Element("dishonest intention", (r"\bstole\b", r"\bstolen\b", r"\btheft\b", r"\bsnatch", r"திருட", r"திருடி", r"பறித்த", r"எடுத்துச் சென்ற")),
        Element("moveable property", (r"\bphone\b", r"\bmobile\b", r"\bcash\b", r"\bmoney\b", r"\bjewel", r"\bbike\b", r"\bbicycle\b", r"\bvehicle\b", r"\bwallet\b", r"\bpurse\b", r"\bgold\b", r"\bchain\b", r"போன", r"மொபைல்", r"செல்", r"பணம்", r"நகை", r"பைக்", r"செயின்"),
                "identify the item taken"),
        Element("taken from possession of another", (r"\bmy\b", r"\bfrom (?:my|his|her|the complainant)", r"\bcomplainant'?s\b", r"என்", r"எங்கள்", r"என்னிடம்", r"கடையில்", r"வீட்டில்"),
                "whose possession was it in?"),
        Element("without consent", (r"\bwithout (?:my|his|her|our)? ?(?:consent|permission|knowledge)", r"\bunknown (?:man|person)", r"\bstole\b", r"அனுமதி இல்லாமல்", r"தெரியாமல்", r"திருடி"),
                "was anything taken with consent, e.g. a loan?"),
    ],
    "380": [  # theft in dwelling
        Element("theft", (r"\bstole\b", r"\bstolen\b", r"\btheft\b", r"திருட", r"திருடி")),
        Element("from a dwelling house / building", (r"\bhouse\b", r"\bhome\b", r"\bshop\b", r"\bbuilding\b", r"\bflat\b", r"\broom\b", r"வீடு", r"வீட்டில்", r"கடை", r"கடையில்"),
                "was the place a dwelling, tent or vessel used as a residence?"),
    ],
    "302": [  # murder
        # actual-death words only: கொலை / "murder" also appear in *threats*
        # ("கொலை செய்வேன்" = I will kill you), which is 506, not 302. Those cues
        # belong to the intent element below.
        Element("death caused", (r"\bdead\b", r"\bdied\b", r"\bdeath\b", r"\bkilled\b", r"\bmurdered\b", r"இறந்த", r"இறப்பு", r"கொன்ற", r"மரணம்", r"கொலை செய்யப்பட்ட", r"உயிரிழந்த"),
                "is the victim deceased?"),
        Element("act done with intention to cause death / knowledge it would", (r"\bmurder", r"\bintention", r"\bdeliberate", r"\bstabbed\b", r"\bstrangl", r"\bshot\b", r"\bpoison", r"கொலை", r"குத்தி", r"கழுத்தை", r"சுட்ட", r"விஷம்"),
                "intention or knowledge: weapon, manner, prior threats?"),
    ],
    "304B": [  # dowry death
        Element("death of a woman", (r"\bdeath of (?:a |the )?(?:woman|wife|daughter|deceased)", r"\b(?:she|wife|daughter|woman|deceased) (?:was found |)(?:dead|died)", r"\bdied\b", r"\bdead\b", r"இறந்த", r"மரணம்", r"தூக்கில்"),
                "is the deceased a woman?"),
        Element("by burns, bodily injury, or otherwise than in normal circumstances", (r"\bburn", r"\binjur", r"\bhang", r"\bpoison", r"\bsuicide", r"\bunnatural", r"\bmurder", r"தீக்குளி", r"தூக்கு", r"விஷம்", r"காயம்", r"தற்கொலை"),
                "cause of death: burns, injury, hanging, poison?"),
        Element("within seven years of marriage", (r"\bmarried (?:\w+ )?(?:\d|one|two|three|four|five|six|seven) years? (?:ago|back)", r"\b(?:\d|one|two|three|four|five|six|seven) years? (?:of|after|since) (?:the |their |her )?marriage", r"\bnewly married", r"திருமணமாகி", r"திருமணம் ஆன"),
                "date of marriage vs date of death"),
        Element("cruelty or harassment for dowry, soon before death", (r"\bdowry", r"\bcruelty", r"\bharass", r"\bdemand", r"\btortur", r"வரதட்சணை", r"கொடுமை", r"துன்புறுத்த", r"சீர்"),
                "dowry demand by husband or his relatives, soon before death"),
    ],
    "498A": [  # cruelty by husband or relatives
        Element("accused is husband or relative of husband", (r"\bhusband\b", r"\bmother-in-law\b", r"\bfather-in-law\b", r"\bin-laws?\b", r"\bhis (?:mother|father|brother|sister|family|relatives?)", r"கணவர்", r"கணவன்", r"மாமியார்", r"மாமனார்", r"புகுந்த வீட்டார்"),
                "relationship of each accused to the woman"),
        Element("cruelty: conduct likely to drive to suicide / grave injury, or harassment for dowry", (r"\bcruelty", r"\bharass", r"\bdowry", r"\bbeat", r"\btortur", r"\bthreat", r"\babus", r"\bdemand", r"கொடுமை", r"வரதட்சணை", r"அடித்த", r"துன்புறுத்த", r"மிரட்ட"),
                "specific acts, dates, and whether demands were for dowry"),
    ],
    "323": [  # voluntarily causing hurt
        Element("hurt caused (bodily pain, disease or infirmity)", (r"\bhit\b", r"\bbeat", r"\bslap", r"\bpunch", r"\bkick", r"\binjur", r"\bassault", r"\bhurt\b", r"\bwound", r"அடித்த", r"அடித்தார்", r"காயம்", r"தாக்கி", r"குத்தி"),
                "nature of injury; medical record?"),
        Element("voluntarily (intention or knowledge)", (r"\bdeliberate", r"\bintention", r"\battack", r"\bassault", r"\bhit\b", r"\bbeat", r"அடித்த", r"தாக்கி"),
                "was it accidental?"),
    ],
    "506": [  # criminal intimidation
        Element("threat of injury to person, reputation or property", (r"\bthreat", r"\bthreaten", r"\bkill you\b", r"\bwill kill\b", r"\bintimidat", r"\bwarn", r"மிரட்ட", r"மிரட்டினார்", r"கொன்று விடுவேன்", r"அச்சுறுத்த"),
                "what exactly was threatened, and to whom?"),
        Element("intent to cause alarm / compel an act or omission", (r"\bthreat", r"\bif (?:you|i|we) (?:do|don't|report|tell)", r"\bforce", r"\bcompel", r"மிரட்ட", r"சொன்னால்", r"பயமுறுத்த"),
                "purpose of the threat"),
    ],
    "420": [  # cheating and dishonestly inducing delivery of property
        Element("deception / cheating", (r"\bcheat", r"\bfraud", r"\bdeceiv", r"\bfake\b", r"\bfalse promise", r"\bpromis", r"\bdup", r"\bscam", r"\botp\b", r"\bunauthori[sz]ed", r"\bclaiming (?:he|she|they|to be)\b", r"\bposing as\b", r"மோசடி", r"ஏமாற்ற", r"பொய்", r"ஓடிபி", r"போலி"),
                "what false representation was made?"),
        Element("dishonest inducement to deliver property", (r"\bgave\b", r"\bpaid\b", r"\btransferred\b", r"\btook (?:rs|₹|money|\d)", r"\btaken from (?:my|the|his|her) (?:account|bank|card)", r"\bdebited\b", r"\bwithdrawn\b", r"\bdelivered\b", r"\binvested\b", r"கொடுத்த", r"பணம் கொடுத்", r"அனுப்பி", r"ரூபாய்", r"கணக்கில் இருந்து", r"எடுக்கப்பட்ட", r"பணம் போ"),
                "what was delivered, when, and how?"),
    ],
    "354": [  # assault or criminal force to woman with intent to outrage modesty
        Element("assault or criminal force on a woman", (r"\bassault", r"\bgrab", r"\bpush", r"\btouch", r"\bpull", r"\bmolest", r"\bwoman\b", r"\bgirl\b", r"\bher\b", r"பிடித்த", r"தொட்ட", r"இழுத்த", r"பெண்"),
                "physical act and who it was directed at"),
        Element("intent / knowledge of outraging modesty", (r"\bmodesty", r"\bmolest", r"\bindecent", r"\bsexual", r"\bobscene", r"\binappropriat", r"பாலியல்", r"அநாகரிக", r"தவறான நோக்க"),
                "nature of the act and words used"),
    ],
    # --- second batch (2026-09-13): the rest of the common station-house offences.
    # Same discipline: ingredients only, cues specific enough that *all* of them
    # firing together is a real question for the officer, not a keyword echo.
    "392": [  # robbery: theft + violence / fear of instant violence
        Element("theft or extortion", (r"\bstole\b", r"\bstolen\b", r"\brobb", r"\bsnatch", r"\btook (?:away )?(?:my|his|her)", r"\blooted\b", r"திருடி", r"பறித்", r"கொள்ளை", r"வழிப்பறி"),
                "what was taken?"),
        Element("violence, or fear of instant death / hurt / restraint, in doing so", (r"\bknife\b", r"\bat (?:knife|gun)point\b", r"\bthreaten", r"\bhit\b", r"\bbeat", r"\bassault", r"\bpushed\b", r"\bforc", r"\bviolen", r"\bweapon", r"கத்தி", r"மிரட்டி", r"அடித்து", r"தாக்கி", r"வலுக்கட்டாய"),
                "was force used or threatened at the moment of taking?"),
    ],
    "384": [  # extortion
        Element("putting a person in fear of injury", (r"\bthreat", r"\bblackmail", r"\bextort", r"\bintimidat", r"\bfear\b", r"மிரட்ட", r"மிரட்டி", r"அச்சுறுத்த", r"பயமுறுத்த"),
                "what harm was threatened?"),
        Element("dishonestly inducing delivery of property / valuable security", (r"\bdemand(?:ed|ing)? (?:rs|₹|\d|money|lakh|payment)", r"\bpay (?:him|them|rs|₹|\d)", r"\bextort", r"\bransom\b", r"\bhafta\b", r"\bmamool", r"\bmoney\b", r"பணம் கேட்", r"ரூபாய் கேட்", r"மாமூல்", r"கப்பம்"),
                "what was demanded, and was anything handed over?"),
    ],
    "406": [  # criminal breach of trust
        Element("property entrusted to the accused", (r"\bentrust", r"\bgave (?:him|her|them) (?:my|the|rs|₹|\d)", r"\bhanded over\b", r"\bdeposit", r"\bfor safe ?keeping", r"\bin (?:his|her|their) custody", r"\bchit\b", r"\bagent\b", r"\btreasurer\b", r"ஒப்படைத்", r"நம்பி கொடுத்", r"சீட்டு", r"பொறுப்பில்"),
                "on what terms was the property entrusted?"),
        Element("dishonest misappropriation or conversion", (r"\bmisappropriat", r"\bdid not return", r"\bnot return", r"\brefus(?:ed|es|ing) to return", r"\bswindl", r"\bused (?:it|the money) for", r"\bconvert", r"\bran away with", r"\bcheat", r"திருப்பி(?:த்)? தர", r"திருப்பித் தரவில்லை", r"கையாடல்", r"ஏமாற்ற", r"தர மறு"),
                "was the property used contrary to the terms?"),
    ],
    "411": [  # dishonestly receiving stolen property
        Element("property was stolen property", (r"\bstolen\b", r"\bstole\b", r"\btheft\b", r"திருட", r"திருடிய"),
                "is the property shown to be stolen?"),
        Element("received / retained knowing or believing it stolen", (r"\breceiv", r"\bbought\b", r"\bpurchas", r"\bfound (?:with|in the possession of)", r"\bin (?:his|her|their) possession", r"\bpawn", r"\bsold (?:it |them )?to\b", r"வாங்கி", r"விற்ற", r"அடகு", r"வைத்திருந்த"),
                "did the receiver know or have reason to believe it was stolen?"),
    ],
    "341": [  # wrongful restraint
        Element("voluntary obstruction of a person", (r"\bblock", r"\bobstruct", r"\bstopped (?:me|him|her|us|them)\b", r"\bwould not let (?:me|him|her|us|them) (?:go|pass|leave)", r"\bdid not (?:let|allow) (?:me|him|her|us|them) (?:to )?(?:go|pass|leave)", r"\bsurround", r"\bcornered\b", r"வழி(?:யை)? மறித்", r"தடுத்", r"போக விடவில்லை", r"சூழ்ந்து"),
                "what act prevented movement?"),
        Element("so as to prevent proceeding in a direction the person had a right to go", (r"\b(?:go|pass|leave|proceed|move|walk|drive)\b", r"\bway\b", r"\broad\b", r"\bpath\b", r"\bgate\b", r"\bdoor\b", r"\bexit\b", r"வழி", r"சாலை", r"ரோடு", r"வாசல்", r"கதவு", r"போக"),
                "where was the person trying to go?"),
    ],
    "342": [  # wrongful confinement
        Element("wrongful restraint", (r"\bconfin", r"\block(?:ed)? (?:me|him|her|us|them|in|inside|up)\b", r"\bkept (?:me|him|her|us|them) (?:in|inside|locked)", r"\bdetain", r"\bheld (?:me|him|her|us|them) (?:in|inside|captive)", r"\bnot allowed to leave\b", r"\bwould not let (?:me|him|her|us|them) (?:out|leave)", r"பூட்டி", r"அடைத்து", r"வெளியே விடவில்லை", r"சிறை"),
                "was the person prevented from leaving?"),
        Element("prevented from going beyond certain limits (a room, house, vehicle)", (r"\broom\b", r"\bhouse\b", r"\bhome\b", r"\boffice\b", r"\bcar\b", r"\bvan\b", r"\bgodown\b", r"\bshed\b", r"\bpremises\b", r"\bfarmhouse\b", r"அறை", r"வீட்டி", r"காரி", r"வேனி", r"கிடங்கு"),
                "where, and for how long?"),
    ],
    "363": [  # kidnapping
        Element("taking or enticing a minor / person of unsound mind, or conveying a person beyond India", (r"\bkidnap", r"\babduct", r"\btook (?:away )?(?:my |the |our )?(?:son|daughter|child|boy|girl|baby|kid)\b", r"\btaken away\b", r"\bentic", r"\blured\b", r"\bmissing\b", r"கடத்த", r"கடத்தி", r"தூக்கிச் சென்ற", r"காணாமல்", r"அழைத்துச் சென்ற"),
                "how was the child taken or enticed?"),
        Element("out of the keeping of the lawful guardian, without consent", (r"\bson\b", r"\bdaughter\b", r"\bchild\b", r"\bminor\b", r"\bboy\b", r"\bgirl\b", r"\bbaby\b", r"\b(?:aged|age) (?:\d|1[0-7])\b", r"\b(?:\d|1[0-7])[ -]years?[ -]old\b", r"\bwithout (?:my|our|the parents'?) (?:consent|permission|knowledge)", r"மகன்", r"மகள்", r"குழந்தை", r"சிறுவ", r"சிறுமி", r"பையன்", r"பெண் குழந்தை"),
                "age of the person taken; guardian's consent?"),
    ],
    "366": [  # kidnapping / abducting woman to compel marriage or force illicit intercourse
        Element("kidnapping or abduction of a woman", (r"\bkidnap", r"\babduct", r"\btook (?:her|my (?:daughter|sister|wife)) (?:away|forcibly|by force)", r"\bforcibly took\b", r"\btaken away\b", r"கடத்த", r"கடத்தி", r"தூக்கிச் சென்ற", r"வலுக்கட்டாயமாக அழைத்து"),
                "was the woman taken by force or deceit?"),
        Element("intent to compel marriage or force / seduce to illicit intercourse", (r"\bmarr", r"\bforc(?:ed|ing)? (?:her )?to marry", r"\bagainst her will", r"\bsexual", r"\billicit", r"\brape", r"திருமணம்", r"கல்யாணம்", r"கட்டாய", r"பாலியல்", r"விருப்பமின்றி"),
                "purpose of the abduction as stated by the victim or family"),
    ],
    "324": [  # voluntarily causing hurt by dangerous weapons or means
        Element("hurt caused voluntarily", (r"\bhit\b", r"\bbeat", r"\bstab", r"\bcut\b", r"\bslash", r"\bassault", r"\battack", r"\binjur", r"\bwound", r"\bhurt\b", r"அடித்", r"குத்தி", r"வெட்டி", r"தாக்கி", r"காயம்"),
                "nature of injury; medical record?"),
        Element("by a dangerous weapon or means (instrument of cutting/stabbing/shooting, fire, poison, acid...)", (r"\bknife\b", r"\bknives\b", r"\bsickle\b", r"\baruval\b", r"\bblade\b", r"\bsword\b", r"\biron (?:rod|pipe|bar)\b", r"\brod\b", r"\bstick\b", r"\blathi\b", r"\bstone\b", r"\bbottle\b", r"\bacid\b", r"\bfire\b", r"\bpetrol\b", r"\bgun\b", r"\bpistol\b", r"\bshot\b", r"\bpoison", r"\bboiling\b", r"கத்தி", r"அருவாள்", r"கம்பி", r"கம்பு", r"கல்", r"பாட்டில்", r"ஆசிட்", r"அமிலம்", r"பெட்ரோல்", r"துப்பாக்கி", r"விஷம்"),
                "what weapon or means was used?"),
    ],
    "325": [  # voluntarily causing grievous hurt
        Element("hurt caused voluntarily", (r"\bhit\b", r"\bbeat", r"\bstab", r"\bassault", r"\battack", r"\binjur", r"\bwound", r"அடித்", r"குத்தி", r"வெட்டி", r"தாக்கி", r"காயம்"),
                "nature of the act"),
        Element("hurt is grievous (fracture, loss of sight/hearing/limb, disfiguration, 15+ days in pain / unable to follow ordinary pursuits, danger to life)", (r"\bfractur", r"\bbroke(?:n)? (?:his|her|my|the) (?:arm|leg|hand|jaw|nose|skull|bone|ribs?|teeth|tooth)", r"\bbroken bone", r"\bgrievous", r"\bserious(?:ly)? injur", r"\bunconscious", r"\bicu\b", r"\bhospitali[sz]ed", r"\badmitted (?:in|to) (?:the )?hospital", r"\bstitch", r"\blost (?:an |his |her |my )?(?:eye|sight|hearing|teeth|tooth|finger)", r"\bdisfigur", r"\bperman", r"\bcritical", r"எலும்பு (?:முறி|உடை)", r"முறிந்த", r"படுகாயம்", r"மயக்க", r"மருத்துவமனையில் அனுமதி", r"தையல்", r"பார்வை இழ", r"கண் இழ"),
                "medical certificate: which clause of 'grievous hurt' applies?"),
    ],
    "307": [  # attempt to murder
        Element("act done with intention / knowledge that, if it caused death, would be murder", (r"\btried to kill", r"\battempt(?:ed)? to (?:kill|murder)", r"\bstab", r"\bshot at\b", r"\bfired at\b", r"\bstrangl", r"\bpoison", r"\bpushed (?:me|him|her) (?:into|off|from|in front of)", r"\bset (?:me|him|her|fire)", r"\bslit\b", r"\bthroat\b", r"கொல்ல முயற்சி", r"கொலை முயற்சி", r"கொல்ல (?:வந்த|பார்த்த)", r"கழுத்தை (?:அறுக்க|நெரித்)", r"குத்தி", r"சுட்ட", r"விஷம்", r"தீ வைத்"),
                "what act, and what shows intent to kill rather than to hurt?"),
        Element("victim survived", (r"\bsurviv", r"\bescaped\b", r"\bhospital", r"\btreatment\b", r"\binjur", r"\bwound", r"\badmitted\b", r"\bsaved\b", r"\balive\b", r"\bI\b", r"\bme\b", r"உயிர் தப்பி", r"காயம்", r"மருத்துவமனை", r"சிகிச்சை", r"தப்பி", r"என்னை", r"நான்"),
                "the victim is alive: 307, not 302"),
    ],
    "306": [  # abetment of suicide
        Element("suicide committed", (r"\bsuicide\b", r"\bkilled (?:him|her)self\b", r"\bhang(?:ed|ing) (?:him|her)self\b", r"\btook (?:his|her) (?:own )?life\b", r"\bconsumed poison\b", r"\bended (?:his|her) life\b", r"தற்கொலை", r"தூக்கிட்டு", r"தூக்கில் தொங்கி", r"விஷம் (?:குடித்|அருந்தி)", r"உயிரை மாய்த்"),
                "cause of death: suicide?"),
        Element("abetment: instigation, conspiracy or intentional aid before the act", (r"\bharass", r"\btortur", r"\bhumiliat", r"\bthreat", r"\binstigat", r"\bprovok", r"\bforc", r"\bdrove (?:him|her) to\b", r"\bbecause of (?:his|her|their|the)\b", r"\bdemand", r"\bloan\b", r"\bdowry", r"\babus", r"கொடுமை", r"துன்புறுத்", r"அவமான", r"மிரட்ட", r"தூண்டி", r"காரணமாக", r"கடன்", r"வரதட்சணை"),
                "what conduct of the accused led to the suicide, and how close in time?"),
    ],
    "279": [  # rash driving on a public way
        Element("driving a vehicle on a public way", (r"\bdriv", r"\brid(?:e|ing|er)\b", r"\bcar\b", r"\bbike\b", r"\bbus\b", r"\blorry\b", r"\btruck\b", r"\bauto\b", r"\bvan\b", r"\btwo[- ]wheeler\b", r"\bscooter\b", r"\bvehicle\b", r"\btempo\b", r"\btractor\b", r"ஓட்டி", r"ஓட்டிய", r"கார்", r"பைக்", r"பஸ்", r"லாரி", r"ஆட்டோ", r"வேன்", r"வாகன"),
                "which vehicle, who was driving?"),
        Element("rashly or negligently, so as to endanger life / be likely to cause hurt", (r"\brash", r"\bnegligen", r"\bspeed", r"\brecklessly?\b", r"\bdrunk", r"\bwrong side\b", r"\bsignal\b", r"\bhit (?:me|him|her|us|the|a|my)\b", r"\bknocked (?:down|me|him|her)", r"\bram+ed\b", r"\bdashed\b", r"\bcollid", r"\baccident\b", r"\bran over\b", r"வேகமாக", r"அதிவேக", r"கவனக்குறைவ", r"மோதி", r"இடித்", r"விபத்து", r"குடித்து", r"தவறான பக்க"),
                "what was rash or negligent about the driving?"),
    ],
    "304A": [  # causing death by negligence
        Element("death caused", (r"\bdied\b", r"\bdead\b", r"\bdeath\b", r"\bkilled\b", r"\bsuccumbed\b", r"\bpassed away\b", r"இறந்த", r"மரணம்", r"உயிரிழந்", r"பலியா"),
                "is the victim deceased?"),
        Element("by a rash or negligent act not amounting to culpable homicide", (r"\brash", r"\bnegligen", r"\baccident", r"\bspeed", r"\bcareless", r"\bran over\b", r"\bknocked down\b", r"\bcollid", r"\bhit (?:him|her|the|a|my)\b", r"\belectric", r"\bshock\b", r"\bopen (?:drain|pit|manhole)", r"\bfell (?:into|from|off)\b", r"\bcollaps", r"\bwithout (?:safety|precaution|barricad)", r"விபத்து", r"கவனக்குறைவ", r"அஜாக்கிரதை", r"வேகமாக", r"மோதி", r"இடித்", r"மின்சார", r"விழுந்து", r"பாதுகாப்பு இல்லாமல்"),
                "what act, and what made it rash or negligent?"),
    ],
    "427": [  # mischief causing damage
        Element("destruction or damage to property", (r"\bdamag", r"\bdestroy", r"\bbroke\b", r"\bbroken\b", r"\bsmash", r"\bvandal", r"\bset (?:fire|on fire|ablaze)", r"\bburnt?\b", r"\bcut (?:the |down )?(?:trees?|crops?|wires?|cables?)", r"\bpunctur", r"\bscratch", r"\btore\b", r"\bdemolish", r"சேதப்படுத்", r"சேதம்", r"உடைத்", r"நொறுக்கி", r"தீ வைத்", r"எரித்", r"அழித்"),
                "what was damaged and how?"),
        Element("wrongful loss caused, with intent or knowledge (value Rs 50 or more under IPC; BNS grades by amount)", (r"\bworth\b", r"\bvalue", r"\bloss\b", r"\bcost", r"\brs\.?\s?\d", r"₹", r"\blakh", r"\bthousand", r"\bdeliberate", r"\bintention", r"\bpurpose", r"\bknowing", r"மதிப்பு", r"ரூபாய்", r"ரூ", r"இழப்பு", r"லட்சம்", r"ஆயிரம்", r"வேண்டுமென்றே", r"திட்டமிட்டு"),
                "value of the damage; was it deliberate?"),
    ],
    "447": [  # criminal trespass
        Element("entry into or upon property in another's possession", (r"\benter", r"\bentered\b", r"\btrespass", r"\bcame into (?:my|our|the)\b", r"\bbarged\b", r"\bintrud", r"\bencroach", r"\boccupied\b", r"\bbroke into\b", r"நுழைந்", r"அத்துமீறி", r"ஆக்கிரமி", r"உள்ளே வந்", r"புகுந்"),
                "whose possession, and was entry permitted?"),
        Element("intent to commit an offence or to intimidate, insult or annoy the possessor", (r"\bthreat", r"\babus", r"\bassault", r"\bdamag", r"\bstole\b", r"\btheft\b", r"\bintimidat", r"\binsult", r"\bannoy", r"\bwithout (?:my|our) (?:permission|consent)", r"\brefus(?:ed|ing) to leave", r"\bquarrel", r"\bfenc", r"\bwall\b", r"\bland\b", r"\bplot\b", r"\bfield\b", r"மிரட்ட", r"திட்டி", r"அடித்", r"சேதப்", r"திருடி", r"அனுமதி இல்லாமல்", r"வெளியேற மறு", r"நிலம்", r"நிலத்தில்", r"தோட்டத்தில்", r"வேலி", r"சுவர்"),
                "purpose of the entry"),
    ],
    "448": [  # house-trespass
        Element("criminal trespass", (r"\btrespass", r"\benter", r"\bentered\b", r"\bbroke into\b", r"\bbarged\b", r"\bintrud", r"\bcame into (?:my|our|the)\b", r"\bforced (?:his|her|their) way\b", r"நுழைந்", r"அத்துமீறி", r"உள்ளே வந்", r"புகுந்", r"கதவை உடைத்"),
                "was the entry unlawful?"),
        Element("into a building used as a human dwelling, place of worship or custody of property", (r"\bhouse\b", r"\bhome\b", r"\bflat\b", r"\bapartment\b", r"\bbedroom\b", r"\bkitchen\b", r"\bshop\b", r"\btemple\b", r"\bchurch\b", r"\bmosque\b", r"\boffice\b", r"\bgodown\b", r"\bwarehouse\b", r"வீடு", r"வீட்டி", r"வீட்டுக்குள்", r"கடை", r"கோவில்", r"கோயில்", r"மசூதி", r"அலுவலக", r"கிடங்கு"),
                "what kind of building?"),
    ],
    "509": [  # word, gesture or act intended to insult the modesty of a woman
        Element("word, sound, gesture, exhibition, or intrusion on privacy, directed at a woman", (r"\bwoman\b", r"\bgirl\b", r"\blady\b", r"\bher\b", r"\bwife\b", r"\bdaughter\b", r"\bsister\b", r"\bmother\b", r"\bme\b", r"பெண்", r"பெண்ணை", r"என்னை", r"மனைவி", r"மகள்", r"சகோதரி"),
                "who was the target?"),
        Element("intended to insult her modesty (obscene words/gestures, stalking, peeping, indecent messages)", (r"\bobscene", r"\bvulgar", r"\blewd", r"\bindecent", r"\bsexual(?:ly)? (?:colou?red|explicit|abus)", r"\bdirty (?:words|messages|comments)", r"\bwhistl", r"\bgestur", r"\bstalk", r"\bfollow(?:ed|ing) (?:me|her)\b", r"\bpeep", r"\bstar(?:ed|ing) at\b", r"\b(?:sent|sends|sending) (?:me |her )?(?:obscene|vulgar|dirty|nude|indecent)", r"\bmodesty", r"\bteas", r"\bcatcall", r"ஆபாச", r"அசிங்கமா", r"கேவலமா", r"கிண்டல்", r"பின் தொடர்", r"சைகை", r"விசில்", r"மானபங்க", r"அவமானப்படுத்"),
                "exact words / acts, and where"),
    ],
    "294": [  # obscene acts and songs in a public place
        Element("obscene act, or obscene song / words, causing annoyance to others", (r"\bobscene", r"\bvulgar", r"\blewd", r"\bindecent", r"\bnaked\b", r"\bnude\b", r"\burinat", r"\bexpos", r"\bfilthy", r"\bdirty (?:words|songs|language)", r"\babus(?:ed|ing|ive) (?:me|us|them|language|words)", r"ஆபாச", r"அசிங்க", r"கெட்ட வார்த்தை", r"திட்டி", r"நிர்வாண", r"கேவலமா"),
                "what act or words, and who was annoyed?"),
        Element("in or near a public place", (r"\bpublic\b", r"\bstreet\b", r"\broad\b", r"\bbus (?:stop|stand)\b", r"\bmarket\b", r"\bpark\b", r"\bshop\b", r"\btemple\b", r"\bschool\b", r"\bcollege\b", r"\bnear (?:my|our|the) (?:house|home)\b", r"\bin front of\b", r"\btheatre\b", r"\bplatform\b", r"\btrain\b", r"\bbus\b", r"பொது", r"தெரு", r"சாலை", r"ரோட்டில்", r"பஸ் ஸ்டாப்", r"மார்க்கெட்", r"கடை", r"கோவில்", r"பள்ளி", r"முன்னால்", r"ரயில்", r"பஸ்"),
                "was the place public?"),
    ],
    "419": [  # cheating by personation
        Element("cheating", (r"\bcheat", r"\bfraud", r"\bdeceiv", r"\bdup", r"\bswindl", r"\bscam", r"\botp\b", r"\bdebited\b", r"\btaken from (?:my|the|his|her) (?:account|bank|card)", r"\bunauthori[sz]ed", r"மோசடி", r"ஏமாற்ற", r"ஏமாந்", r"ஓடிபி", r"கணக்கில் இருந்து"),
                "what was the deception?"),
        Element("by pretending to be some other person, or representing oneself as someone one is not", (r"\bpretend(?:ed|ing)? to be\b", r"\bposing as\b", r"\bposed as\b", r"\bimpersonat", r"\bpersonat", r"\bclaim(?:ed|ing)? to be\b", r"\bintroduced himself as\b", r"\bfake (?:id|identity|profile|officer|police|official|call|account)", r"\bin the name of\b", r"\bsaid (?:he|she) was (?:from|a|an|the)\b", r"\bcalled (?:me )?(?:saying|claiming) (?:he|she|they) (?:was|were|is|are) from\b", r"\b(?:bank|police|customs|cbi|income tax|army|government) (?:officer|official|staff|employee|manager)\b", r"\bkyc\b", r"\botp\b", r"போலி", r"போல் நடித்", r"என்று கூறி", r"என்று சொல்லி", r"அதிகாரி என்று", r"பேங்க் என்று", r"வங்கி(?:யில்)? இருந்து (?:பேசு|அழை)", r"ஓடிபி", r"போலீஸ் என்று"),
                "who did the accused claim to be?"),
    ],
}

_COMPILED: dict[str, list[tuple[Element, re.Pattern]]] = {
    sec: [(el, re.compile("|".join(f"(?:{c})" for c in el.cues), re.IGNORECASE)) for el in els]
    for sec, els in ELEMENTS.items()
}


def covered_sections() -> list[str]:
    return sorted(ELEMENTS)


def check_elements(ipc_section: str, narrative: str, utterance_id: str | None = None) -> list[ElementCheck]:
    """Element checks for `ipc_section` against `narrative`.

    Returns [] for sections without an element list. Otherwise one ElementCheck
    per element: `yes` with the matched cue(s) quoted and their offsets, or
    `unclear` with the note saying what to establish. Never `no`.
    """
    spec = _COMPILED.get(ipc_section.strip().upper())
    if not spec:
        return []
    out: list[ElementCheck] = []
    for el, pattern in spec:
        hits = list(pattern.finditer(narrative))
        if hits:
            quotes = []
            prov = []
            seen: set[str] = set()
            for m in hits[:3]:
                q = m.group(0)
                if q.lower() in seen:
                    continue
                seen.add(q.lower())
                quotes.append(q)
                prov.append(Provenance(utterance_id=utterance_id,
                                       transcript_char_start=m.start(),
                                       transcript_char_end=m.end()))
            out.append(ElementCheck(
                element=el.name,
                satisfied="yes",
                justification="narrative says: " + "; ".join(f"'{q}'" for q in quotes),
                supporting_provenance=prov,
            ))
        else:
            out.append(ElementCheck(
                element=el.name,
                satisfied="unclear",
                justification=("not stated in the narrative" + (f" -- establish: {el.note}" if el.note else "")),
            ))
    return out


@dataclass(frozen=True, slots=True)
class CueHit:
    """A covered section whose ingredients the narrative evidences."""

    ipc_section: str
    evidenced: int
    total: int
    cues: tuple[str, ...]          # the matched words, for the officer

    @property
    def fraction(self) -> float:
        return self.evidenced / self.total if self.total else 0.0


def scan_all(narrative: str, min_fraction: float = 1.0) -> list[CueHit]:
    """Every covered section whose elements are evidenced in `narrative` at or
    above `min_fraction` (default: all of them), best first.

    This is the **recall safety net** for the classifier. The classifier reads
    an English view of the complaint; when that view comes from machine
    translation, legally decisive words can vanish ("வரதட்சணை" -> "relief"
    happened on the first real run, and a dowry-cruelty complaint routed to CSR).
    The cue lists are bilingual and run on the *original* transcript, so they
    survive that. A section with all its ingredients present in the original but
    absent from the prediction is not a finding -- it is a question for the
    officer, and it is raised as one.
    """
    hits: list[CueHit] = []
    for sec, spec in _COMPILED.items():
        matched: list[str] = []
        n_yes = 0
        for _el, pattern in spec:
            m = pattern.search(narrative)
            if m:
                n_yes += 1
                matched.append(m.group(0))
        if spec and n_yes / len(spec) >= min_fraction:
            hits.append(CueHit(sec, n_yes, len(spec), tuple(dict.fromkeys(matched))))
    hits.sort(key=lambda h: (-h.fraction, -h.total, h.ipc_section))
    return hits


def summarise(checks: list[ElementCheck]) -> str:
    """'3/4 elements evidenced' for a one-line view."""
    if not checks:
        return "elements not analysed"
    yes = sum(1 for c in checks if c.satisfied == "yes")
    return f"{yes}/{len(checks)} elements evidenced"
