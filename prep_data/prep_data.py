import os
import datetime
from datasets import load_dataset

DEFAULT_DATASET = "arc_easy,arc_challenge,winogrande,piqa,openbookqa,hellaswag"

def _naive_length_filtering(text, min_length=200, max_length = 10000000):
    text_length = len(text.split())
    
    if text_length < min_length:
        return False
    if text_length > max_length:
        return False
    return True

def _naive_date_filtering(date, min_date='2020-01-01', max_date='2022-01-01'):
    date = date.split('T')[0]
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d')
    max_date = datetime.datetime.strptime(max_date, '%Y-%m-%d')
    
    if date < min_date:
        return False
    if date > max_date:
        return False
    return True

def ultrachat(train_dataset_length, test_dataset_length):
    calibration_train_dataset = load_dataset("HuggingFaceH4/ultrachat_200k",
                                        cache_dir='./DATA',
                                        )['train_sft']['messages'][:train_dataset_length]
    calibration_train_dataset = [i[:2] for i in calibration_train_dataset]
    
    calibration_test_dataset = load_dataset("HuggingFaceH4/ultrachat_200k",
                                        cache_dir='./DATA',
                                        )['test_sft']['messages'][:test_dataset_length]
    calibration_test_dataset = [i[:2] for i in calibration_test_dataset]
    
    return calibration_train_dataset, calibration_test_dataset
    
def helpsteer2(train_dataset_length, test_dataset_length):
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    calibration_train_dataset = load_dataset("nvidia/HelpSteer2",
                                        cache_dir='./DATA',
                                        )['train']
    calibration_train_dataset.filter(lambda example: 3 < ((example['helpfulness']>3) +
                                                    (example['correctness']>3) +
                                                    (example['coherence']>3) +
                                                    (example['complexity']>3) +
                                                    (example['verbosity']>3)))
    calibration_train_dataset = calibration_train_dataset.select(range(train_dataset_length))
    calibration_train_dataset = [[{'content': i['prompt'], 'role': 'user'}, {'content': i['response'], 'role': 'assistant'}] for i in calibration_train_dataset]
    
    calibration_test_dataset = load_dataset("nvidia/HelpSteer2",
                                        cache_dir='./DATA',
                                        )['validation']
    calibration_test_dataset.filter(lambda example: 3 < ((example['helpfulness']>3) +
                                                    (example['correctness']>3) +
                                                    (example['coherence']>3) +
                                                    (example['complexity']>3) +
                                                    (example['verbosity']>3)))
    calibration_test_dataset = calibration_test_dataset.select(range(test_dataset_length))
    calibration_test_dataset = [[{'content': i['prompt'], 'role': 'user'}, {'content': i['response'], 'role': 'assistant'}] for i in calibration_test_dataset]
    
    return calibration_train_dataset, calibration_test_dataset

def wikipedia_used(train_dataset_length, test_dataset_length):
    calibration_dataset = load_dataset("SamuelYang/wikipedia_20200501.en",
                                        cache_dir='./DATA',
                                        )['train'].filter(lambda example: _naive_length_filtering(example['text']))
    calibration_train_dataset = calibration_dataset[:train_dataset_length]['text']
    
    calibration_test_dataset = calibration_dataset[train_dataset_length:train_dataset_length+test_dataset_length]['text']
    
    return calibration_train_dataset, calibration_test_dataset

def fineweb_unused(train_dataset_length, test_dataset_length):
    calibration_dataset = load_dataset("HuggingFaceFW/fineweb", name='sample-10BT',
                                        cache_dir='./DATA',
                                        )['train'].filter(lambda example: (example['language_score'] > 0.90) and _naive_length_filtering(example['text']) and _naive_date_filtering(example['date']))['text']
    
    calibration_train_dataset = calibration_dataset[:train_dataset_length]
    calibration_test_dataset = calibration_dataset[train_dataset_length:train_dataset_length+test_dataset_length]    
    
    return calibration_train_dataset, calibration_test_dataset

def lorem_ipsum():
    raw_text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla at justo ac enim lacinia convallis ac nec est. Integer cursus sapien velit, et tincidunt leo ultricies at. Morbi quis enim pretium, pulvinar est vitae, tincidunt justo. Fusce rhoncus mi ante. In hac habitasse platea dictumst. Ut sapien nisl, eleifend sit amet felis et, posuere elementum velit. Donec risus ex, tincidunt non erat tempor, suscipit aliquam ante. Suspendisse potenti. In vehicula mauris eu gravida porta. Vestibulum dictum risus blandit felis pellentesque, ut efficitur tellus sodales. Nam imperdiet vestibulum dolor vel mattis. Duis rhoncus ut mauris id eleifend. Mauris laoreet facilisis quam, et malesuada ante sodales vel. Nulla libero velit, viverra ac viverra eget, tempus ut odio.

    Aliquam ultrices placerat felis, sit amet pulvinar erat ultrices nec. Sed suscipit urna quis tempor semper. Aenean mattis quam vitae risus maximus, non auctor ipsum efficitur. Curabitur at fringilla purus, et faucibus ipsum. Donec et urna varius, eleifend orci eu, maximus nunc. Nullam pharetra aliquet purus quis pretium. Suspendisse ipsum augue, ullamcorper quis diam sit amet, pulvinar pulvinar felis. Sed venenatis quis lacus non euismod. Aenean cursus venenatis odio, ac semper lorem congue vel. Sed ac lectus id urna consequat feugiat.

    Proin a rhoncus eros. Morbi a ultricies metus, eget consectetur enim. Curabitur a consectetur lorem. Vestibulum volutpat tempus dolor. Nullam pretium posuere sem fringilla imperdiet. Donec finibus est eu sapien feugiat, et lobortis tellus ultricies. Phasellus sit amet est arcu.

    Proin mollis convallis sapien sed blandit. Sed tristique purus in diam pretium, a euismod purus tempor. Proin egestas laoreet magna, sit amet pretium lectus accumsan et. Sed laoreet sit amet libero et posuere. Praesent eu scelerisque nisl. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Praesent nec porttitor velit, ac porttitor sem.

    Sed elementum metus nec elit ullamcorper sodales. Phasellus in ante eget nisi pretium accumsan ut in sem. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Sed vel velit mollis, pharetra nibh id, malesuada ligula. Donec sem augue, laoreet vel maximus ac, luctus quis enim. Donec ut ligula nec sapien suscipit fringilla a ac dolor. Fusce rutrum sodales lectus. Suspendisse condimentum sagittis commodo. Phasellus consectetur tellus ex, sodales pretium diam convallis vel. Quisque blandit condimentum volutpat. Nullam nec ipsum massa. Morbi sodales, dolor in scelerisque iaculis, massa augue luctus dui, sit amet consequat turpis libero ut urna.

    Maecenas ut est a nibh dignissim luctus. Sed non malesuada erat. Vivamus mollis sapien sed nibh dictum scelerisque a et diam. Mauris mattis sodales lobortis. Integer fringilla suscipit augue. Sed ut nibh lacinia ligula gravida tincidunt sit amet a nisi. Suspendisse pellentesque aliquet ipsum, ac convallis diam luctus vitae. Suspendisse porttitor nulla ut erat tristique sagittis. Morbi pulvinar, augue et dictum pharetra, elit nibh dictum ante, nec dapibus dolor tellus ut est. Maecenas dignissim, sem dignissim elementum dignissim, lacus dui ullamcorper dui, eget semper orci tellus id dui. Etiam magna nisi, mattis quis orci ac, scelerisque facilisis justo.

    Nulla non mauris ac turpis tempus suscipit. Curabitur gravida vel lorem non venenatis. Phasellus euismod at augue faucibus aliquam. Suspendisse in posuere dolor. Nullam nisl ligula, pellentesque sed rhoncus congue, feugiat vitae tellus. Nullam molestie hendrerit nulla vel suscipit. Donec auctor turpis ut mi dictum efficitur.

    In id massa ante. Praesent cursus pretium vehicula. Fusce quis nulla ipsum. Vivamus sed lectus sagittis, molestie enim eget, scelerisque nibh. Vivamus vel nulla ut lectus sagittis auctor. Morbi porttitor arcu a nibh semper, eu sollicitudin ante scelerisque. Quisque luctus efficitur rhoncus. Vivamus tortor ipsum, faucibus quis hendrerit nec, imperdiet at tellus. Integer quis felis in lorem eleifend feugiat. Quisque finibus, quam at fringilla euismod, sem nibh bibendum elit, at iaculis ex nunc vitae erat. Aliquam lacus erat, blandit a volutpat ac, ornare sit amet justo. Fusce ut blandit ipsum, et aliquam nisl. Nunc at nisl eget lectus faucibus scelerisque. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus.

    Cras at neque nisl. Nullam ante libero, venenatis ac auctor ac, tempor consequat felis. Aenean lacus tellus, aliquam sit amet nisl eget, porttitor ultricies eros. Mauris id ultricies nisi, eget imperdiet enim. Vestibulum vitae quam quam. Nulla consequat a augue id volutpat. Etiam fringilla cursus viverra. Praesent fermentum, mi non maximus lacinia, eros dolor condimentum leo, et sagittis ante felis a ipsum.

    Etiam et nisl ex. Nulla facilisi. Integer leo ipsum, volutpat ac suscipit vel, maximus eu nibh. Sed vitae risus sit amet nulla malesuada maximus. Aliquam at urna vitae eros pellentesque auctor. In hac habitasse platea dictumst. Aenean eu viverra neque, a imperdiet velit. Nulla lacus quam, mattis ac sapien quis, fringilla condimentum ligula. Cras pellentesque, lacus sit amet eleifend viverra, arcu massa pharetra tortor, ut tincidunt lacus sapien nec libero. Ut ullamcorper euismod neque, sit amet lobortis odio cursus nec. Donec venenatis at nisi eu tempus. Vestibulum ut odio eget velit dictum posuere. Curabitur feugiat, tortor eu tincidunt ornare, lacus lacus ullamcorper enim, vitae sollicitudin nisl felis a ligula. Integer non arcu justo.

    Pellentesque dignissim ante semper, pellentesque odio vitae, mollis leo. Etiam et lorem urna. Morbi pulvinar felis id orci tincidunt bibendum. Nulla pretium, nunc sed egestas hendrerit, ante urna viverra risus, et mattis arcu urna id ligula. Nunc tellus ipsum, tempus non metus quis, posuere commodo leo. Curabitur et malesuada velit. Duis ullamcorper massa eget lectus suscipit, a malesuada lectus dapibus. Sed sed lorem quis velit tincidunt posuere. Donec vel tortor ante. Vivamus ornare urna et lacus varius, facilisis lacinia urna sollicitudin. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Pellentesque gravida pellentesque dui. Praesent dui sapien, sagittis et ullamcorper sed, maximus lobortis turpis. Phasellus erat urna, sagittis nec turpis vel, auctor rutrum tellus.

    Integer quis sapien sit amet leo imperdiet bibendum. Praesent id dictum est, sed efficitur nisi. Ut vitae lectus quis lorem rhoncus posuere at vitae lectus. Donec pretium non ipsum ac molestie. Morbi at risus at velit ultricies vehicula. Aliquam erat volutpat. Aliquam eget finibus ipsum. Suspendisse pellentesque luctus tortor a euismod. Aliquam euismod justo nec feugiat vehicula. Cras efficitur condimentum dictum. In sit amet bibendum metus. Nam tempus, erat ac hendrerit venenatis, justo nunc facilisis nibh, vitae dapibus libero purus quis velit. Aenean venenatis est id magna eleifend, a efficitur purus luctus. Nam id ipsum nec diam molestie maximus eu a nisi.

    In quis bibendum purus. Nam ac sollicitudin quam. Donec eu massa eu nisi iaculis facilisis. Maecenas sed tortor orci. Nunc eu dolor consectetur, hendrerit nunc eget, aliquet lectus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Donec a elementum urna. Curabitur velit nisi, condimentum eu nisi vel, tincidunt maximus nulla.

    Cras imperdiet varius suscipit. In ac erat mauris. Pellentesque aliquam hendrerit massa vitae consequat. Cras vitae gravida dui. Proin faucibus sem eget aliquam auctor. Suspendisse potenti. Ut cursus arcu eu condimentum porttitor.

    Nam sed gravida tellus. Etiam ut dignissim ex. Sed eu ex nec tellus pretium convallis. In mollis consequat vestibulum. Integer a imperdiet justo. Etiam velit tellus, pulvinar sed lacinia id, ornare non nibh. Vestibulum pretium turpis et auctor ultricies. Nulla in diam est.

    Nunc eu ligula ultrices, tincidunt diam ac, auctor ex. Etiam in suscipit lectus. Proin elementum in ante id lobortis. Etiam viverra vel augue quis aliquam. Suspendisse lectus purus, pulvinar nec felis ut, blandit convallis leo. Suspendisse convallis sollicitudin lectus ut tempor. Vivamus id arcu et tortor rhoncus semper ut ac dolor. Aliquam lacus tellus, posuere et lorem ut, consectetur luctus magna. Quisque eu elit magna. Maecenas nisi justo, aliquam vel ligula imperdiet, laoreet pretium neque. Etiam at pharetra odio, nec accumsan tellus. Aenean non ipsum tortor. Aliquam facilisis ullamcorper augue, a luctus quam pellentesque sed. Sed ullamcorper nisi ac dui maximus ultrices.

    Sed vestibulum ex in porta accumsan. Phasellus congue justo in velit viverra, in maximus dolor volutpat. Pellentesque elementum sit amet dolor non commodo. In quis iaculis ex. Fusce a euismod est, ut sollicitudin odio. Quisque efficitur augue tellus, non efficitur augue dictum in. Pellentesque lobortis quam arcu, ut sagittis lectus congue nec. In tincidunt velit facilisis suscipit tempus. Etiam sed odio in tortor ullamcorper ullamcorper. Donec mauris dolor, volutpat non posuere ut, volutpat id odio. In hac habitasse platea dictumst. Vestibulum hendrerit dictum mi, id rutrum ante finibus eget. Donec mattis quis massa aliquet aliquam. Vestibulum sit amet efficitur massa.

    Quisque congue, diam eu ullamcorper pretium, libero metus semper nisl, quis porta elit augue pretium ligula. Morbi tincidunt risus aliquet, rutrum libero sed, dapibus eros. In sit amet orci vel justo vehicula varius. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam et ultrices elit. Etiam et iaculis quam, quis blandit ligula. Sed convallis tellus risus, ac eleifend dolor feugiat et.

    In et condimentum tortor, commodo mollis elit. Curabitur dictum leo quis mollis faucibus. Maecenas dictum erat sit amet massa hendrerit egestas. Interdum et malesuada fames ac ante ipsum primis in faucibus. Pellentesque vehicula tempus imperdiet. Donec malesuada orci et mattis accumsan. Nulla fermentum, ante et efficitur mollis, enim sapien dapibus purus, non imperdiet erat libero nec turpis. Fusce ut luctus ex, vitae pellentesque augue. Etiam lacinia lorem non felis ultrices elementum.

    Etiam efficitur odio eu metus eleifend iaculis. Praesent posuere mauris et nulla feugiat, eget accumsan nunc vestibulum. Vestibulum ac commodo nulla, nec fringilla eros. Integer dolor magna, sodales vitae justo non, lobortis pharetra nibh. Vivamus accumsan arcu vitae leo consequat, ac molestie nunc faucibus. Vestibulum vitae sagittis neque. Aenean eleifend sagittis turpis vitae vestibulum. Integer euismod enim sem, eget tincidunt nisi porttitor ac. Duis dictum dignissim volutpat. Integer id egestas nulla. Sed sagittis cursus nibh quis volutpat. Integer ante ipsum, finibus in arcu porta, laoreet pharetra lectus. Aliquam ipsum massa, efficitur id dapibus vel, ultrices eu magna. Fusce ut pharetra dui. Sed non lacus purus. Duis et tellus nisl.

    Maecenas ultricies, metus mattis rhoncus rutrum, metus risus vestibulum lacus, eu elementum purus ex in magna. Duis ante sem, molestie vitae ullamcorper non, aliquet sit amet elit. Ut accumsan augue nec turpis mollis, ut interdum quam ultrices. Aliquam erat volutpat. Curabitur placerat magna aliquam, imperdiet lorem eget, aliquam diam. Aenean sollicitudin eros quis suscipit volutpat. Curabitur ac lacus malesuada, porta lectus sit amet, consectetur dui. Nulla a pellentesque magna. Phasellus sodales elementum felis, sed imperdiet velit aliquet eu. Donec eget imperdiet lacus. Morbi egestas ut risus eget posuere. Vestibulum sagittis fringilla facilisis. Mauris pulvinar, justo in dictum fermentum, neque sem rhoncus ante, sit amet malesuada elit ante in libero. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; In hac habitasse platea dictumst.

    Proin aliquet molestie tincidunt. Donec hendrerit mi non lobortis vehicula. Phasellus eu hendrerit lorem. Integer gravida nibh eu mattis dictum. Vestibulum posuere eros eget tortor tempor commodo. Praesent bibendum ullamcorper eros, a vulputate ipsum consequat quis. Sed volutpat feugiat nulla, vel dictum urna sollicitudin eget. Pellentesque enim urna, lobortis eu nisl quis, commodo hendrerit justo. In quis tortor malesuada sem placerat tempus. Fusce sit amet facilisis lacus.

    Vivamus efficitur eu magna eu tristique. Maecenas mi felis, imperdiet id malesuada quis, consectetur maximus sapien. Suspendisse potenti. Cras nec magna eu eros vulputate molestie. Mauris lacus libero, maximus sit amet justo at, hendrerit commodo nisi. Phasellus et dignissim tortor. Aliquam at eros ultricies, maximus mauris eu, semper sapien. Sed aliquam iaculis turpis. Pellentesque accumsan ex felis, et tristique purus finibus in. Suspendisse pellentesque urna in lacinia molestie.

    Nulla eget egestas ante. Nunc varius magna dolor, a lacinia velit ullamcorper ac. Integer sodales egestas pulvinar. Duis est elit, lacinia in volutpat nec, interdum bibendum dolor. Cras congue lorem quam, quis volutpat enim venenatis egestas. Pellentesque egestas facilisis egestas. Phasellus in justo sit amet lacus imperdiet efficitur. Duis sapien lorem, volutpat in nisi ac, auctor mollis eros. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed viverra mi egestas nulla interdum aliquet. Sed nec cursus nunc.

    Morbi in suscipit nibh. Donec a eros sit amet neque volutpat vehicula vitae in nisl. Vestibulum viverra magna nulla, nec tempus velit hendrerit quis. Maecenas molestie venenatis lectus quis semper. Integer cursus, augue vel mattis aliquet, ligula enim gravida urna, in blandit ex turpis sodales sapien. Ut imperdiet volutpat lectus, id mattis odio feugiat ut. Vestibulum vestibulum tortor id sodales pulvinar. Mauris eget rutrum magna, et finibus justo. Sed sagittis quis sem non bibendum. Suspendisse risus metus, commodo vel fermentum quis, bibendum eget elit. Aliquam a arcu dui. Donec a turpis ante.

    Curabitur eros mi, porttitor at tempus sed, finibus ut massa. Sed ut massa lectus. Nulla ultrices viverra dignissim. Pellentesque eu egestas magna. Donec accumsan libero non viverra rhoncus. Nam et nulla auctor, tincidunt velit nec, lobortis erat. Morbi quis lacinia libero. Curabitur mattis ac velit ut dignissim. Etiam sit amet tristique lorem. Sed mollis dolor augue.

    In hac habitasse platea dictumst. Cras posuere ultricies felis, quis ullamcorper nisi congue vitae. Vivamus elementum elit eget lorem faucibus, quis convallis urna fringilla. Suspendisse orci tellus, faucibus id sagittis sed, efficitur euismod arcu. Fusce sagittis elementum dolor, in eleifend lorem malesuada non. Aliquam erat volutpat. Duis rutrum nulla ut arcu tincidunt vulputate. Etiam fermentum ligula vitae molestie volutpat. In odio felis, sodales non dui sed, venenatis feugiat magna. Vestibulum malesuada ipsum sit amet metus efficitur, eu lobortis ex tempus. Sed ultrices congue arcu non fermentum. In porta bibendum gravida. Nunc ullamcorper ante sed nisl venenatis, vitae consectetur ligula faucibus. Vivamus pellentesque nulla et dapibus tempor.

    Etiam metus neque, placerat nec condimentum ut, semper ut neque. Phasellus sagittis risus vel sapien vulputate, at suscipit nulla elementum. Mauris mi tellus, dictum efficitur massa auctor, commodo aliquet urna. Pellentesque id turpis lectus. In volutpat imperdiet lorem, vel lacinia eros mattis eu. Nunc condimentum finibus nulla, consequat vulputate lectus hendrerit eu. Morbi pretium, tellus vitae sodales dapibus, nibh urna sollicitudin libero, vel dignissim tortor justo tempus felis. Aenean venenatis sapien turpis, at dictum nisl pharetra eget. Proin nunc purus, auctor vitae imperdiet dignissim, laoreet non ligula. Quisque placerat nisl et tellus convallis, sit amet porttitor lacus aliquet. In hac habitasse platea dictumst. Integer cursus semper nunc, vel euismod urna egestas eu.

    Proin non ipsum augue. Ut ut nisi vel arcu sagittis fringilla eu vel diam. Mauris ultricies, metus non vestibulum tempor, eros elit pretium nisl, sit amet laoreet eros mi vel felis. Aenean volutpat dui non aliquet rutrum. Morbi laoreet lacus malesuada felis venenatis consectetur. Sed odio velit, faucibus non pharetra quis, fermentum et lectus. Quisque eleifend dolor at velit volutpat porttitor. Maecenas vehicula dignissim sapien et aliquam. Quisque rutrum nec elit sit amet pharetra. Proin mattis faucibus magna facilisis tristique. In eget arcu eget neque aliquam imperdiet at id nisl. Ut justo lorem, tincidunt at facilisis vel, mattis id libero. Cras ac feugiat sapien, quis euismod quam.

    Maecenas sit amet ipsum nisi. Curabitur dignissim tincidunt libero, ac hendrerit nisi faucibus sit amet. Vivamus a lobortis nulla. Nulla dignissim dui quam, eu fermentum tellus facilisis in. Nam ut nunc dictum, tristique purus eu, dictum ex. Maecenas quis purus sit amet nunc dapibus porttitor. Vestibulum tristique pretium vulputate. Vivamus nec elit tempus, tincidunt nunc commodo, fermentum velit. Donec non lacus ipsum. Curabitur quis libero feugiat, porta magna a, vestibulum."""
    
    return raw_text, raw_text
    