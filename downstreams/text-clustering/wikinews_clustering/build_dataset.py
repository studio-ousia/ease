from collections import Counter
import fileinput
import re
from tqdm import tqdm
import argparse
tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*?>(?:([^<]*)(<.*?>)?)?')
#                    1     2               3      4
keyRE = re.compile(r'key="(\d*)"')
import json

from WikiExtractor import Extractor

import sys,os
sys.path.append(os.path.abspath("/home/fmg/nishikawa/EASE"))
from utils.sentence_tokenizer import MultilingualSentenceTokenizer
# nl sv th ko ca sr fr pl eo sd fa li ar no "fi" ro cs bs it de sq ta hu ja es bg he el ru en pt zh tr uk  

def pages_from(input, catMark):
    catRE = re.compile(fr'\[\[{catMark}([^\|]+).*\]\].*')  

    """
    Scans input extracting pages.
    :return: (id, revid, title, namespace key, page), page is a list of lines.
    """
    page = []
    id = None
    ns = '0'
    last_id = None
    revid = None
    inText = False
    redirect = False
    title = None
    for line in input:
        if not isinstance(line, str): line = line.decode('utf-8')
        if '<' not in line:  # faster than doing re.search()
            if inText:
                page.append(line)
                # extract categories
                if line.lstrip().startswith(f'[[{catMark}'):
                    mCat = catRE.search(line)
                    if mCat:
                        catSet.add(mCat.group(1))
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
            catSet = set()
            redirect = False
        elif tag == 'id' and not id:
            id = m.group(3)
        elif tag == 'id' and not revid:
            revid = m.group(3)
        elif tag == 'title':
            title = m.group(3)
        elif tag == 'ns':
            ns = m.group(3)
        elif tag == 'redirect':
            redirect = True
        elif tag == 'text':
            if m.lastindex == 3 and line[m.start(3)-2] == '/': # self closing
                # <text xml:space="preserve" />
                continue
            inText = True
            line = line[m.start(3):m.end(3)]
            page.append(line)
            if m.lastindex == 4:  # open-close
                inText = False
        elif tag == '/text':
            if m.group(1):
                page.append(m.group(1))
            inText = False
        elif inText:
            page.append(line)
        elif tag == '/page':
            if id != last_id and not redirect:
                yield (id, revid, title, ns,catSet, page)
                last_id = id
                ns = '0'
            id = None
            revid = None
            title = None
            page = []

def clean(extractor, text):
    text = extractor.transform(text)
    text = extractor.wiki2text(text)
    text = extractor.clean(text)
    text = text.replace("\n", "")
    return text

def main():

    parser = argparse.ArgumentParser(description="このプログラムの説明（なくてもよい）")  # 2. パーサを作る
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--input_path", type=str, default="/home/fmg/nishikawa/corpus/wikinews")
    # parser.add_argument("--output_path", type=str, default="/home/fmg/nishikawa/EASE/text-clustering/data/wikinews")
    parser.add_argument("--output_path", type=str, default="/home/fmg/nishikawa/EASE/text-clustering/data/label-unified-wikinews")
    parser.add_argument("--min_text_length", type=int, default=5)
    args = parser.parse_args()

    input_file = f"{args.input_path}/{args.lang}wikinews-20211101-pages-articles.xml.bz2"
    file = fileinput.FileInput(input_file, openhook=fileinput.hook_compressed)

    with open('/home/fmg/nishikawa/EASE/text-clustering/wikinews_clustering/en_cat_to_lang_cat.json') as f:
        en_cat_to_lang_cat = json.load(f)


    def lang_to_mainCat(lang):
        mainCat = []

        if lang == "en":
            return set(en_cat_to_lang_cat.keys())
        for enCatName in en_cat_to_lang_cat.keys():
            if lang in en_cat_to_lang_cat[enCatName]:
                mainCat.append(en_cat_to_lang_cat[enCatName][lang])
        return set(mainCat)

    # lang_to_mainCat = {
    #     "en": {"Crime and law", "Culture and entertainment", "Disasters and accidents", "Economy and business","Education", "Environment", "Health", "Obituaries", "Politics and conflicts", "Science and technology", "Sports", "Wackynews", "Local only", "Media", "Weather", "Women"},
    #     "ar": {"قانون وجرائم", "بيئة" , "نقل", "صحة", "اقتصاد", "رياضة", "علوم وتقنية", "سياسة", "ثقافة", "وفيات", "كوارث وحوادث", "تربية وتعليم"},
    #     "ja": {"政治", "経済", "社会", "文化", "スポーツ", "学術", "ひと", "気象", "脇ニュース"},
    #     "es": {"Arte, cultura y entretenimiento", "Ciencia y tecnología", "Clima", "Deportes", "Desastres y accidentes", "Ecología", "Economía y Negocios", "Judicial", "Mundo loco", "Obituario", "Política", "Salud", "Sociedad"},
    #     "tr": {"Afetler ve kazalar", "Bilim ve Teknoloji", "Ekonomi ve iş", "Eğitim", "Hava durumu", "Kültür ve eğlence", "Medya", "Politika", "Spor", "Suç ve hukuk", "Sağlık", "Terörizm", "Ulaşım", "Yıllarına göre konular", "Çevre"},
    #     "it": {"Ambiente", "Argomenti principali", "Argomenti secondari", "Cultura e società", "Curiosità", "Disastri e incidenti", "Economia e finanza", "Giustizia e criminalità", "Meteo", "Necrologi", "Politica e conflitti", "Scienza e tecnologia", "Società", "Sport", "Trasporti"},
    #     "ko": {"경제", "과학기술", "국제", "날씨", "문화", "부고", "사고", "사회", "성명", "세계", "소방", "스포츠", "연예", "정치"},
    #     "pt": {"Ciência e tecnologia", "Controvérsias", "Crime, Direito e Justiça", "Cultura e entretenimento", "Desastres e acidentes", "Economia e negócios", "Educação", "Feiras e eventos", "Homem e sociedade", "Militar", "Mistérios", "Música", "Política e conflitos", "Religião", "Saúde", "Sociedade", "Transportes", "Tópico dos dossiês", "Ufologia"},
    #     "ru": {"Интернет", "Культура", "Наука и технологии", "Некрологи", "Общество", "Политика", "Преступность и право", "Происшествия", "Рейтинги", "Религия", "Дни рождения", "Спорт", "Экономика"},
    #     "uk": {"Вікімедіа", "Допомога", "Економіка", "Культура", "Матеріали Maidanua.org", "Матеріали VOA News", "Наука", "Події", "Політика", "Світ", "Спорт", "Суспільство", "IT"},
    #     "cs": {"Ekonomika", "Katastrofy", "Politika", "Počasí", "Společnost", "Věda a technika"},
    #     "pl": {"Gospodarka", "Kalendarium", "Katastrofy i klęski żywiołowe", "Kultura i rozrywka", "Nauka", "Polityka", "Prawo i przestępczość", "Religia", "Sport", "Społeczeństwo", "Technika", "Tematy dotyczące Polski", "Tematyczne serie artykułów", "Środowisko", "Świat"},
    #     "ca": {"A la babalà", "Ciència i tecnologia", "Crim i llei", "Cultura i esplai", "Dret", "Drets humans", "Economia", "Espectacle", "Esports", "Insòlit", "Medi ambient", "Necrologia", "Polítiques i conflictes", "Salut", "Societat", "Successos", "Transport"},
    #     "fi": {"Ammattiyhdistysliike", "Avaruus", "Henkilöt", "Ihmisoikeudet", "Kulttuuri ja viihde", "Liikenne", "Media", "Onnettomuudet", "Opetus ja opiskelu", "Politiikka", "Rasismi", "Rikos ja oikeus", "Talous", "Tekniikka", "Terveys", "Tiede ja tekniikka", "Toisiinsa liittyvät uutiset", "Urheilu", "Ympäristö"},
    #     "fa": {"سیاست و منازعات", "فرهنگ و سرگرمی", "گردهمایی‌ها", "محیط زیست", "ورزش", "حمل و نقل", "دانش و فناوری","درگذشت‌ها", "رسانه", "زنان", "سلامت", "آب و هوا", "اقتصاد و تجارت", "بلایا و حوادث", "تحصیلات", "جالب", "جرایم و قانون"},
    #     "nl": {"Onderwerpdossiers", "Conflict", "Cultuur", "Economie", "Energie", "Gebouw", "Geografie", "Geschiedenis", "Koningshuis", "Levenscyclus", "Mens en maatschappij", "Natuur", "Organisatie", "Politiek", "Ramp", "Recht", "Recreatie", "Religie", "Tijd", "Veiligheid", "Verkeer en vervoer", "Wetenschap", "Wikimedia in het nieuws"},
    #     "hu": {"Balesetek és katasztrófák", "Egészség és életmód", "Gazdaság", "Halálozások", "Jog és bűnügyek", "Kultúra és szórakozás", "Környezet", "Oktatás", "Politika", "Sport", "Tudomány és technika", "Társadalom"},
    #     "eo": {"Akcidentoj", "Amaskomunikiloj", "Ekonomio", "Homoj", "Juro", "Katastrofoj", "Kulturo", "Mediprotektado", "Organizaĵoj", "Politiko", "Rangolistoj", "Scienco", "Scienco kaj teknologio", "Socio", "Sporto", "Teknologio", "Vetero"},
    #     # "hu": {"", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""},
    #     # "hu": {"", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""},
    # }

    lang_to_catMark = {
        "en": "Category:",
        "ar": "تصنيف:",
        "ja": "Category:",
        "es": "Categoría:",
        "tr": "Kategori:",
        "it": "Categoria:",
        "ko": "분류:",
        "pt": "Categoria:",
        "ru": "Категория:",
        "uk": "Категорія:",
        "cs": "Kategorie:",
        "pl": "Kategoria:",
        "ca": "Categoria:",
        "fi": "Luokka:",
        "fa": "رده:",
        "nl": "Categorie:",
        "hu": "Kategória:",
        "eo": "Kategorio:", 
        'bg': 'Категория:', # 単数形: Категория   Категории  Категория
        'de': 'Kategorie:', # 単数形: Kategorie Kategorien
        'fr': 'Catégorie:',
        'sv': 'Kategori:',
    }

    # mainCat = lang_to_mainCat[args.lang]
    mainCat = lang_to_mainCat(args.lang)
    catMark = lang_to_catMark[args.lang]
    cnt = 0
    org_titles = []
    org_categories = []
    org_texts = []
    sentence_tokenizer = MultilingualSentenceTokenizer(args.lang)

    for page_data in tqdm(pages_from(file, catMark)):
        id, revid, title, ns, catSet, page = page_data

        candCat = None
        if len(catSet) > 0:
            # check catRE 
            cnt += 1
        for cat in catSet:
            if cat in mainCat:
                if candCat == None:
                    candCat = cat
                
                # filter out multiple categories
                else:
                    candCat = None
                    break
        
        if candCat:
            # cnt += 1
            ex = Extractor(id, revid, title, page)
            org_text = clean(ex, ex.text)
            if len(org_text) < args.min_text_length: continue
            
            org_titles.append(ex.title)
            org_categories.append(candCat)
            org_texts.append(org_text)

    titles = []
    texts = []
    categories = []
    first_sentences = []
    cnt_dic = Counter(org_categories)

    # filter out minor classes
    for title, category, text in zip(org_titles, org_categories, org_texts):
        if cnt_dic[category] >= 10:
            titles.append(title)
            texts.append(text)
            categories.append(category)
            try:
                first_sentences.append(sentence_tokenizer.tokenize(text)[0])
            except:
                first_sentences.append("")


    title_and_texts = [title + text for title, text in zip(titles, texts)]
    title_and_first_sentences = [title + sent for title, sent in zip(titles, first_sentences)]

    print("include mainCat num: ", cnt)

    print("dataset num: ", len(titles))
    print("class num: ", len(cnt_dic))

    path = f"{args.output_path}/{args.lang}_titles.txt"
    with open(path, mode='w') as f:
        f.write('\n'.join(titles))

    path = f"{args.output_path}/{args.lang}_texts.txt"
    with open(path, mode='w') as f:
        f.write('\n'.join(texts))

    path = f"{args.output_path}/{args.lang}_title_and_texts.txt"
    with open(path, mode='w') as f:
        f.write('\n'.join(title_and_texts))

    path = f"{args.output_path}/{args.lang}_sentences.txt"
    with open(path, mode='w') as f:
        f.write('\n'.join(first_sentences))

    path = f"{args.output_path}/{args.lang}_title_and_sentences.txt"
    with open(path, mode='w') as f:
        f.write('\n'.join(title_and_first_sentences))
        
    path = f"{args.output_path}/{args.lang}_categories.txt"
    with open(path, mode='w') as f:
        f.write('\n'.join(categories))

if __name__ == "__main__":
    main()
