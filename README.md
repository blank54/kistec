# KISTEC X C!LAB PROJECT
<!-- 
## Configuration
'./thesaurus' 폴더 하위에 다음의 파일이 필요함
* stop_list.txt
* synonyms.txt
* userword.txt


## Import Library
    sys.path.append(../kistec/)
    from preprocess import MyPreprocess
    mp = MyPreprocess()
    
## Preprocess
### Apply synonyms
    sent = '배수 시설은 배수시설로 배수관도 배수시설로 배수만 있는 건 그대로'
    sent_synonym = mp.synonym(sent)
    print(sent_synonym)

### Stopword Removal
    sent_stop = mp.stopword_removal(sent_synonym, return_type='str') # Default return type: list
    print(sent_stop)

### Customized PoS tagging
    from konlpy.tag import Komoran

    mp.build_userdic('./thesaurus/userword.txt')
    komoran = Komoran(userdic=mp.fname_userdic) # './thesaurus/userdic.txt'

    print(sent)
    print(komoran.nouns(sent))
    print(komoran.nouns(sent_synonym))
    print(komoran.nouns(sent_stop)) -->