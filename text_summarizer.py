import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large")

# Function to clean the input text
def clean_text(text):
    # Remove speaker labels and timestamps using regex
    clean_text = re.sub(r"Speaker SPEAKER_\d+ \(\d+\.\d+s - \d+\.\d+s\):", "", text)
    clean_text = re.sub(r"\(\d+\.\d+s - \d+\.\d+s\)", "", clean_text)  # Remove remaining standalone timestamps
    return clean_text.strip()

# Function to summarize cleaned text with repetition control
def summarize(text):
    # Preprocess the input text for summarization
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary with repetition handling parameters
    summary_ids = model.generate(input_ids, 
                                 max_length=250,  # Adjust max length to control the length of the summary
                                 min_length=100,  # Ensure a minimum length for a more detailed summary
                                 length_penalty=1.0,  # Neutral length penalty
                                 num_beams=6,  # More beams for better output exploration
                                 no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
                                 repetition_penalty=2.0,  # Penalize repeated tokens to avoid redundancy
                                 early_stopping=True)
    
    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example usage
text = """
Speaker SPEAKER_02 (0.45s - 0.89s):  Hi.
Speaker SPEAKER_02 (1.06s - 5.84s):  I'm Chai Hovelenia, I will be your moderator for the panel this afternoon.
Speaker SPEAKER_02 (6.04s - 16.50s):  And I'd like to call to the stage our two mayors, Matromanila mayors, we're very lucky to have them, 
Mayor Joy Belmonte of Casson City.
Speaker SPEAKER_02 (18.91s - 21.26s):  Mayor Iskaw Morano of Manila.
Speaker SPEAKER_02 (24.53s - 24.90s):  Thank you.
Speaker SPEAKER_02 (27.40s - 30.24s):  and Paolo Alcazarin.
Speaker SPEAKER_02 (31.40s - 37.65s):  Urban planner who I'm sure you've heard about and you've read his pieces.
Speaker SPEAKER_02 (41.27s - 54.40s):  Okay, so this afternoon what will be tackling will be making metro areas livable. This is something 
that I know is very close to your heart as it is to mine because every single day we feel that
Speaker SPEAKER_02 (54.87s - 59.06s):  things are getting worse and they're not getting better. So.
Speaker SPEAKER_02 (59.26s - 73.88s):  I'd like to introduce a little more formally our panelists. Mayor Joy Belmonte served as vice mayor 
of Cassandra City for nine long years before she finally became mayor.
Speaker SPEAKER_02 (74.11s - 75.88s):  She was elected this year.
Speaker SPEAKER_02 (76.17s - 76.74s):  Um,
Speaker SPEAKER_02 (77.17s - 98.95s):  Mayor Belmontis Advocacies have focused on the protection and upholding of rights of women and children, gender fairness, economic empowerment of women, mental health, culture and the arts. Also, please welcome in the flesh, not just on Facebook Live, Mayor Isco Moreno.
Speaker SPEAKER_02 (99.91s - 120.62s):  He has been promoting open governance, and I think this is the reason why you see him really on Facebook. That's part of the effort. He also recently led clearing operations in Manila and also announced his commitment to preserve and expand the Aroseros Forest Park.
Speaker SPEAKER_02 (125.60s - 160.07s):  And finally, let's welcome our urban planning expert, Paulo Alcazaran. He has been a practicing design consultant for the last 38 years. That's almost four decades. Some of those years were spent in Singapore. Paulo's advocacies include 
heritage, conservation, green cities, bringing back usable sidewalks, parks, and plazas to our cities. If you're familiar with the Iloilo Esplanade, I don't know if some of you are from Iloilo here.
Speaker SPEAKER_02 (160.17s - 167.36s):  and the pedestrian and park networks of Makati and Oritigas, that's Paolo's handy work.
Speaker SPEAKER_02 (167.82s - 172.27s):  Okay, so let's move on to our discussion this afternoon.
Speaker SPEAKER_02 (173.46s - 191.36s):  As I said earlier, I'm sure that you feel the congestion. You feel that this city is about to explode. And the mayors, I'm sure, also have their plates very full. And part of the reality is they only have the ears.
Speaker SPEAKER_02 (191.83s - 205.45s):  One term is good for three years. Of course, they can seek re-election and be in office for as long as nine years. Maybe I'd like to start by asking, what are your priorities given?
Speaker SPEAKER_02 (205.72s - 215.34s):  given the reality of a short or long-term limit of nine years. What are you prioritizing to make your respective cities more livable?
Speaker SPEAKER_02 (217.80s - 220.42s):  May I call you Scott? I'll see you at Scott.
Speaker SPEAKER_03 (219.02s - 219.44s):  Thank you.
Speaker SPEAKER_03 (220.84s - 221.33s):  Thank you.
Speaker SPEAKER_03 (223.96s - 231.86s):  Thank you. Magandang hapunusainyo sa mga kasam ako. Well, Manila,
Speaker SPEAKER_03 (232.10s - 233.41s):  It's very unique.
Speaker SPEAKER_03 (233.80s - 240.52s):  comparatively speaking with our neighboring cities, with regard to hosting
Speaker SPEAKER_03 (241.97s - 243.45s):  all the major ports.
Speaker SPEAKER_03 (245.14s - 251.37s):  We are serving as part of the JME area.
Speaker SPEAKER_03 (252.23s - 256.09s):  in the south, in the sparras.
Speaker SPEAKER_03 (257.21s - 257.75s):  ¡Tarlán!
Speaker SPEAKER_03 (258.74s - 259.65s):  in the north.
Speaker SPEAKER_03 (260.24s - 262.86s):  with regard to their role model.
Speaker SPEAKER_03 (263.50s - 270.15s):  being delivered to them by our ports, our ports and the west side of the city.
Speaker SPEAKER_03 (271.09s - 271.79s):  easily.
Speaker SPEAKER_03 (272.21s - 273.44s):  in a given time.
Speaker SPEAKER_03 (274.57s - 275.26s):  in the day.
Speaker SPEAKER_03 (275.92s - 278.20s):  four thousand moon boxes
Speaker SPEAKER_03 (278.33s - 280.78s):  Ross Boulibard, Finance.
Speaker SPEAKER_03 (281.17s - 283.60s):  Lăton, porci în plăton.
Speaker SPEAKER_03 (284.31s - 287.61s):  and so on and so forth, especially in the note part of
Speaker SPEAKER_03 (287.97s - 289.76s):  The City, The Artén.
Speaker SPEAKER_03 (290.42s - 290.97s):  uh
Speaker SPEAKER_03 (291.53s - 297.69s):  Then after that in the middle part which is the backbone of the city of Manila, which was beautiful.
Speaker SPEAKER_03 (298.38s - 301.60s):  Just in front of you, tap up in you.
Speaker SPEAKER_03 (301.82s - 306.01s):  all the way to Keson, Bully Bard, and Spain.
Speaker SPEAKER_03 (306.23s - 310.43s):  This is our backbone with regard to our roads.
Speaker SPEAKER_03 (311.71s - 314.02s):  ay yung mga ngay lang an.
Speaker SPEAKER_03 (314.09s - 316.08s):  n'han pogut tant que és un city.
Speaker SPEAKER_03 (316.61s - 320.81s):  Akyo mga ngayon ng papuntaan para niya, kaya, ay kailangan dumaan na.
Speaker SPEAKER_03 (321.70s - 322.48s):  tough, I've been you.
Speaker SPEAKER_03 (324.32s - 325.08s):  Then nature
Speaker SPEAKER_03 (325.21s - 326.80s):  We have the Northbound Group.
Speaker SPEAKER_03 (326.97s - 329.14s):  bas terminals.
Speaker SPEAKER_03 (330.29s - 332.62s):  that we have to accommodate. These are provincial.
Speaker SPEAKER_03 (333.13s - 334.38s):  bas terminas.
Speaker SPEAKER_03 (334.43s - 334.80s):  No.
Speaker SPEAKER_03 (335.40s - 336.54s):  So technically,
Speaker SPEAKER_03 (337.40s - 339.88s):  It is very easy to deal with our roles.
Speaker SPEAKER_02 (339.98s - 341.21s):  סנטר שתלגה
Speaker SPEAKER_02 (341.90s - 342.56s):
Speaker SPEAKER_03 (342.02s - 344.18s):  We have to...
Speaker SPEAKER_02 (342.59s - 342.64s):  you
Speaker SPEAKER_03 (344.62s - 350.15s):  S�atong instellat ng 의� Korayabacto dito, lwaong fasa setsi Santarat, ay nalacon ko.
Speaker SPEAKER_03 (350.71s - 352.04s):  In this case,
Speaker SPEAKER_03 (352.33s - 355.50s):  MMD and DPWH.
Speaker SPEAKER_03 (356.45s - 357.17s):  But...
Speaker SPEAKER_03 (358.59s - 359.20s):  Ehh
Speaker SPEAKER_03 (360.43s - 362.84s):  wala ako maisip na solution sa trapi.
Speaker SPEAKER_02 (365.36s - 366.25s):
Speaker SPEAKER_03 (367.60s - 369.59s):  Natan Les i Sara Cuvanpier.
Speaker SPEAKER_03 (370.49s - 374.45s):  na tanlesin di ko pa yagano ng mga tagakasun si Tipumunta ng mainila.
Speaker SPEAKER_01 (375.21s - 375.60s):
Speaker SPEAKER_03 (375.70s - 376.19s):  Thank you.
Speaker SPEAKER_01 (376.53s - 377.46s):  Bye Bye
Speaker SPEAKER_03 (377.71s - 378.35s):  아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아
, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아,  
아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아
Speaker SPEAKER_03 (378.98s - 381.07s):  Ребיע Lev CON.
Speaker SPEAKER_03 (381.49s - 381.56s):
Speaker SPEAKER_03 (381.68s - 392.10s):  So, having said that, the least thing that we can do as city government is to clear our streets of all types of obstruction.
Speaker SPEAKER_03 (394.01s - 406.55s):  That's the very least. So if we are going to prioritize things before I develop vertically and duplicate our roads vertically, which is new to impossible because it will cost a lot of money, it will eat a lot of time.
Speaker SPEAKER_00 (395.16s - 395.53s):  Thank you.
Speaker SPEAKER_03 (407.14s - 410.06s):  and I think national government for that matter.
Speaker SPEAKER_03 (410.26s - 411.75s):  should address this.
Speaker SPEAKER_03 (412.29s - 414.46s):  vertical duplication of roads.
Speaker SPEAKER_03 (414.95s - 416.69s):  especially the major one.
Speaker SPEAKER_03 (417.11s - 418.02s):  The backbone.
Speaker SPEAKER_03 (418.58s - 425.91s):  ...and kami na man sa local like see joy, yung nang sigurampin na kamalit na ming bahagi.
Speaker SPEAKER_03 (426.24s - 428.45s):  na pédingagawin ... bila ngl Russell pangaladen!
Speaker SPEAKER_03 (428.99s - 430.68s):  Wagnang Hayaan
Speaker SPEAKER_03 (430.92s - 437.99s):  Pake na bangan ng iilan ang aming mga kalsada, ang kalsada i para salahat.
Speaker SPEAKER_03 (438.51s - 439.35s):  putrafi
Speaker SPEAKER_03 (439.86s - 440.96s):  بھی کلٹ چاپی
Speaker SPEAKER_03 (441.73s - 442.63s):  매일
Speaker SPEAKER_03 (443.10s - 445.55s):  It should serve its purpose.
Speaker SPEAKER_03 (445.78s - 446.34s):  So
Speaker SPEAKER_03 (447.10s - 449.68s):  Prioritization in terms of implementing
Speaker SPEAKER_03 (450.00s - 454.96s):  with regard to addressing traffic and confronting traffic problems.
Speaker SPEAKER_03 (455.01s - 456.60s):  is going back to Beijing.
Speaker SPEAKER_02 (456.94s - 467.11s):  So you're saying that Ampina Khaduwa Ball is clearing operations. So you're clearing operations. Are you, is that the same priority for you, Joy?
Speaker SPEAKER_03 (463.25s - 463.98s):  Thank you.
Speaker SPEAKER_03 (463.99s - 464.63s):  We marry and get Sunny, sweetheart!
Speaker SPEAKER_00 (464.63s - 464.65s):  you
Speaker SPEAKER_00 (467.72s - 468.90s):  with actually man.
Speaker SPEAKER_00 (469.85s - 470.32s):  Thank you.
Speaker SPEAKER_00 (470.47s - 572.63s):  If you're talking about internship livability, I agree of course with my good friend, Mayor Isko, 
and he's my good friend in reality, because he and I were together in the vice mayor's league. So we started both as vice mayor's and now we're reconciled as mayor's. So it's good to have Manila as our neighbor. But in terms of livability, although you asked me earlier what I wanted to do, it's totally different from the issue of traffic, no? But here in Keson City, one thing that we're trying to do, apart from clearing the streets, which is something that was mandated by the president through his state of the nation address, that we, Mayor's in 60 days, must clear primary, secondary, and Mabuhay lanes of all road constructions and sidewalk obstructions. And in these particular areas, 24-7, you can't go through vendors and other areas. So this is what I'm saying, and we're going through a lot of our best. In addition to this, we're doing what we saw in livability is working towards walkable cities. So we're trying to change our land use plan to make sure that land use plan is not only residential, it's not only a work area commercial, but also a mix of use and water places. To make sure that land use plan is well-known, if you can use a bicycle or a vehicle, ideally, with the use of technology, you don't even have to leave your house to be able to complete a day's work. Right.
Speaker SPEAKER_02 (472.97s - 473.68s):  ability.
Speaker SPEAKER_01 (495.30s - 495.62s):  Thanks for watching.
Speaker SPEAKER_02 (572.23s - 580.04s):  I have a quick question. When you say what you're meeting in Sakhyan,
Speaker SPEAKER_02 (580.63s - 583.28s):  Walk if you can, use your bikes if you can.
Speaker SPEAKER_02 (583.79s - 586.89s):  I think you experienced the na me in natin, is that...
Speaker SPEAKER_02 (587.08s - 597.37s):  Number one, pollution. Pag nagbay kako along ed sa, I'm in danger, kasi mababangga ako nga mga bus or malalang hub ko yung maraming hangin.
Speaker SPEAKER_02 (597.76s - 619.29s):  And second, walaang protection for pedestrians. If you walk along many of the main thoroughfairs, 
or even kait yung small streets, para ang walaang mahitang sidewalk, because as we mentioned earlier, vendors have taken over some of these sidewalks. So, popaan nyo, how do you...
Speaker SPEAKER_02 (619.60s - 622.62s):  How do you address that? Even if I want to?
Speaker SPEAKER_02 (623.26s - 623.85s):  But
Speaker SPEAKER_02 (624.36s - 628.22s):  Walaang incentive for me, and it may not be safe also.
Speaker SPEAKER_00 (628.71s - 676.89s):  You ask me within the next three years. So, this is the present. This is the situation you're describing now. Traffic pollution, workplaces away from the home. The need to use public transportation that is inefficient. The need to bring 
your own cars, et cetera. And use ads on other roads that are full of obstructions, et cetera. But I'm thinking in the long term. So in the long term, it is about planning. So it's about changing your comprehensive land use plan. It's about changing the mindset of the people towards walkability, towards more compact cities, if you will. We're in all activities that you need to survive to live day by day can be done within a small area of your city.
Speaker SPEAKER_02 (633.11s - 633.47s): 웃
Speaker SPEAKER_02 (676.89s - 683.16s):  Okay, you mentioned planning, perfect segue to Paolo. What do you say to this?
Speaker SPEAKER_01 (683.96s - 703.11s):  Well, Mayor Joy already touched on all of the important points and so did Mary's go. And I agree within the first three years the best you can do is cleaning and clearing and the projects that we've been involved with in Iloylo and other cities. That's the first thing that can be done and it's doable and it has to be done and sustained.
Speaker SPEAKER_01 (703.70s - 704.11s):  Uh.
Speaker SPEAKER_01 (704.61s - 706.54s):  I do have to remind everyone.
Speaker SPEAKER_01 (707.38s - 708.73s):  that traffic
Speaker SPEAKER_01 (708.92s - 710.27s):  is not the problem.
Speaker SPEAKER_01 (711.18s - 718.65s):  Traffic is just the symptom of the problem of a lack of integrated comprehensive transport system 
for the whole metropolis.
Speaker SPEAKER_01 (720.10s - 720.46s):  Thank you.
Speaker SPEAKER_01 (721.98s - 726.90s):  People always ask me, can you solve traffic? Then that's the wrong question.
Speaker SPEAKER_01 (727.83s - 759.49s):  The thing is to provide Metro Manila in all of our metro areas all around the country with a comprehensive transport system. And that is rail-based mass transport system, non-motorized bikes and walking. And that's where you have to tweak the priorities of the DPWH and all of the national agencies towards that type of infrastructure. We've been involved in a number of initiatives in Makati, in Pasig. And these are for pedestrian systems.
Speaker SPEAKER_01 (760.00s - 779.71s):  except that the cost of them is too small to even get into the notice of the OTR. Most of the projects are sub 500 million. Ang tiniti ng nanilat dahil bill-bill-bill is the multi-billion pass all. All we're asking is
Speaker SPEAKER_01 (779.99s - 808.70s):  One support of a skyway, vertical roads, one supporting structure will build 10 pedestrian bridges. Yet we don't have pedestrian bridges. Just try crossing any of our major thoroughfares. So we need, like Mayor Joyce said, a re-alignment of our priorities from just pure car-based infrastructure to more comprehensive transport systems.
Speaker SPEAKER_02 (809.04s - 815.31s):  Merean bang ganong plans for Manila, the setting up a pedestrian network, a pedestrian system.    
Speaker SPEAKER_03 (816.36s - 817.02s):  وλλى
Speaker SPEAKER_03 (819.04s - 823.58s):  We, I think, architects, so we'll start within our
Speaker SPEAKER_03 (824.21s - 826.16s):  Please hope to go for nuns.
Speaker SPEAKER_03 (826.80s - 829.88s):  because before I go further and ask people around.
Speaker SPEAKER_03 (830.58s - 834.18s):  to participate. We always wanted to show that we
Speaker SPEAKER_03 (835.14s - 836.39s):  in our own house.
Speaker SPEAKER_03 (836.90s - 841.72s):  Dona mein ba yung pakita? We're closing roads for opening more green space.
Speaker SPEAKER_03 (842.38s - 844.96s):  in Lothon.
Speaker SPEAKER_03 (845.40s - 846.90s):  particolare, no?
Speaker SPEAKER_03 (847.38s - 848.54s):  so that...
Speaker SPEAKER_03 (849.47s - 850.70s):  I want yours.
Speaker SPEAKER_03 (851.05s - 852.79s):  masana yan takong lumakan.
Speaker SPEAKER_03 (853.23s - 854.31s):  If you can see...
Speaker SPEAKER_03 (854.34s - 855.36s):  I-I-I-I
Speaker SPEAKER_03 (856.05s - 857.67s):  recently about
Speaker SPEAKER_03 (858.48s - 859.81s):  11 PM.
Speaker SPEAKER_03 (860.74s - 862.29s):  in the evening.
Speaker SPEAKER_03 (862.90s - 864.84s):  tiniting ng ko yung mga utaw.
Speaker SPEAKER_03 (865.31s - 868.20s):  dyo sa tapat ng lotong.
Speaker SPEAKER_03 (869.19s - 870.66s):  binok sa namin yung park.
Speaker SPEAKER_03 (871.47s - 872.57s):  binog sa naman yun.
Speaker SPEAKER_03 (872.79s - 873.18s):  with
Speaker SPEAKER_03 (873.87s - 874.76s):  لیٹرالی
Speaker SPEAKER_03 (875.03s - 876.59s):  No obstruction.
Speaker SPEAKER_03 (877.53s - 879.67s):  literally sin from
Speaker SPEAKER_03 (880.28s - 881.80s):  100% to zero.
Speaker SPEAKER_03 (882.05s - 882.91s):  Obstraktschen.
Speaker SPEAKER_03 (883.88s - 884.42s):  then
Speaker SPEAKER_03 (885.18s - 889.36s):  مین خبخان نیونه نا نا قیمی ها بیتن نتعوره نهی چید بی
Speaker SPEAKER_03 (889.93s - 894.59s):  ...mabagot alo game mindsets. Teni-tinyan naman. Sabi niyo kailangan yun ng...
Speaker SPEAKER_03 (894.68s - 899.45s):  Dada ananya si nesabimo, binaig ang katanan dahanan ng gina gamit mo lo tonparin.
Speaker SPEAKER_03 (899.84s - 903.13s):  Ano dumparin yung bankayatan din beginagamit. Why?
Speaker SPEAKER_03 (903.87s - 904.62s):  I'm your bucket.
Speaker SPEAKER_03 (906.67s - 907.69s):  Yung nakasana yan.
Speaker SPEAKER_02 (909.14s - 909.58s):
Speaker SPEAKER_03 (910.02s - 912.66s):  the typical Filipino creb mentality.
Speaker SPEAKER_03 (914.05s - 914.98s):  Una-unahan.
Speaker SPEAKER_03 (916.70s - 918.12s):  des�이ques-Musiques queические l'onmeurent était une projectingure et oily...
Speaker SPEAKER_03 (918.79s - 922.33s):  Jadi bis dah faltar geng tek datang yang itu Slig Daz, nge-egap
Speaker SPEAKER_03 (922.44s - 923.57s):  towers nalowton?
Speaker SPEAKER_03 (924.11s - 925.22s):  أتمدة لمدة نور
Speaker SPEAKER_03 (925.34s - 926.16s):  Berandil ya?
Speaker SPEAKER_03 (926.87s - 928.21s):  ang makigipag unahan sa Jeep.
Speaker SPEAKER_03 (928.86s - 929.22s):  So.
Speaker SPEAKER_03 (929.94s - 933.02s):  As you can see, real talk, not I, real talk.
Speaker SPEAKER_03 (933.45s - 936.85s):  Paralam natin ang obligasyoninyo ang obligasyonamen.
Speaker SPEAKER_03 (937.29s - 937.67s):  No?
Speaker SPEAKER_03 (940.07s - 940.85s):  people.
Speaker SPEAKER_03 (941.99s - 942.94s):  should participate.
Speaker SPEAKER_03 (944.68s - 946.55s):  especially if you see your government.
Speaker SPEAKER_03 (947.19s - 949.23s):  really moving, having a nerd.
Speaker SPEAKER_03 (949.62s - 951.85s):  και ο第一ος, να έχει μάσpees προς εκμενή.
Speaker SPEAKER_03 (951.92s - 953.23s):  Follow simple rules.
Speaker SPEAKER_03 (953.76s - 954.87s):  That is Jay Mocking.
Speaker SPEAKER_03 (955.90s - 956.96s):  at 11 p.m.
Speaker SPEAKER_03 (958.89s - 959.24s):  See?
Speaker SPEAKER_03 (959.68s - 960.99s):  So if you want to walk,
Speaker SPEAKER_03 (961.26s - 962.61s):  will give you a place to walk.
Speaker SPEAKER_03 (962.80s - 963.73s):  and we're doing it.
Speaker SPEAKER_03 (964.22s - 967.69s):  But are you going to assure me as government?
Speaker SPEAKER_03 (968.37s - 969.65s):  Are you going to walk?
Speaker SPEAKER_03 (970.46s - 971.69s):  in a particular
Speaker SPEAKER_03 (972.23s - 972.94s):  Safe.
Speaker SPEAKER_03 (973.95s - 975.49s):  newly open.
Speaker SPEAKER_03 (976.18s - 977.04s):  space.
Speaker SPEAKER_03 (977.30s - 980.82s):  That is now a question I should ask the people.
Speaker SPEAKER_03 (981.48s - 982.70s):  Are you now ready?
Speaker SPEAKER_03 (983.89s - 985.38s):  because we are ready.
Speaker SPEAKER_03 (986.09s - 988.28s):  We are moving heaven and earth and
Speaker SPEAKER_03 (988.67s - 991.08s):  spending our political capital.
Speaker SPEAKER_03 (992.40s - 993.73s):  to the maximum level.
Speaker SPEAKER_03 (994.54s - 995.89s):  No eeps, no butts.
Speaker SPEAKER_03 (996.89s - 998.79s):  Dos yung walpayulating.
Speaker SPEAKER_03 (999.37s - 1001.14s):  role is in this case.
Speaker SPEAKER_03 (1001.65s - 1004.19s):  occupying not supposed to be occupying.
Speaker SPEAKER_03 (1004.85s - 1006.83s):  will be considered obstruction.
Speaker SPEAKER_03 (1007.55s - 1010.67s):  dan per 줄 panjang προas daripada alkali. Tugin, tawar, masyarakat dan blobir regulator. Baru-baru perstop ber setahun... ...dia hardersukan meletelian. Lihat ini, sampai biasa dan begitu penjakitku yang arent tergalacah... ...cingker, mend 폈 lebih banyak dan lebih memakitkan CCS. Mereka mungkin memperkaitkan ter dalam ruang separuh mempvecksa untuk senjakaan."
Speaker SPEAKER_03 (1011.35s - 1012.53s):  ...will you smartil you?
Speaker SPEAKER_03 (1013.14s - 1013.71s):  My soul.
Speaker SPEAKER_03 (1014.07s - 1014.93s):  o acetilín.
Speaker SPEAKER_03 (1016.85s - 1017.61s):  which we did.
Speaker SPEAKER_03 (1018.50s - 1019.20s):  because they
Speaker SPEAKER_03 (1019.48s - 1019.85s):
Speaker SPEAKER_03 (1020.26s - 1021.14s):  it is really
Speaker SPEAKER_03 (1022.03s - 1022.76s):  They think
Speaker SPEAKER_03 (1023.97s - 1025.17s): 阿atching
Speaker SPEAKER_03 (1025.98s - 1027.04s):  un gobierno
Speaker SPEAKER_03 (1028.73s - 1030.18s):  Somuk trapish
Speaker SPEAKER_03 (1030.96s - 1031.97s):  then they try.
Speaker SPEAKER_03 (1033.07s - 1033.69s):  I think.
Speaker SPEAKER_03 (1034.10s - 1034.89s):  yesterday.
Speaker SPEAKER_03 (1035.31s - 1036.04s):  They failed.
Speaker SPEAKER_03 (1036.92s - 1038.75s):  beko siinase tilin kutalaga.
Speaker SPEAKER_03 (1039.26s - 1041.51s):  sa kasaisaya ng Pilipinas Pison.
Speaker SPEAKER_03 (1042.87s - 1043.72s):  You can see it.
Speaker SPEAKER_03 (1044.68s - 1047.90s):  to show that we really mean business. So, going back.
Speaker SPEAKER_03 (1051.39s - 1052.02s):  ميهراب
Speaker SPEAKER_03 (1052.59s - 1053.25s):  in a way.
Speaker SPEAKER_03 (1053.62s - 1054.77s):  بți buildingzn
Speaker SPEAKER_03 (1055.60s - 1059.93s):  falei сюда للصال أظهر الضعت حط some of you
Speaker SPEAKER_03 (1060.30s - 1061.55s):  habigok, ingat kayo, ah.
Speaker SPEAKER_03 (1061.92s - 1064.74s):  lahat ang istudjante iso, sonud gunas at tap abinyu.
Speaker SPEAKER_03 (1067.54s - 1068.51s):  Sejae Walking-
Speaker SPEAKER_03 (1070.48s - 1072.40s):  ال hanging الدهل free
Speaker SPEAKER_03 (1073.48s - 1075.53s):  Bisa ko may pagitanati mo ng kapataan.
Speaker SPEAKER_03 (1076.93s - 1079.95s):  because you're my dear, our elders.
Speaker SPEAKER_03 (1079.96s - 1082.78s):  wala na meyo sa bingadau anin.
Speaker SPEAKER_03 (1083.34s - 1087.81s):  Mr President, please do you cannot teach all dogs new tricks.
Speaker SPEAKER_03 (1088.52s - 1091.03s):  sort of saying, but we can start among...
Speaker SPEAKER_03 (1091.76s - 1092.84s):  ourselves to you.
Speaker SPEAKER_03 (1092.96s - 1100.06s):  Dan organisasi lewat dan tinggi mushrooms lebih komentar Dan selepas vois hari ini, Ia semakin minuman kamu
Speaker SPEAKER_03 (1100.82s - 1103.59s):  பயண்பம்பம் கேட்டாம் வில்லைக்கும் போகிறேன்.
Speaker SPEAKER_03 (1104.43s - 1105.01s):  traffic.
Speaker SPEAKER_03 (1106.79s - 1108.14s):  will cause a harm.
Speaker SPEAKER_03 (1108.43s - 1109.41s):  will cause a
Speaker SPEAKER_03 (1110.10s - 1110.84s):  Preno.
Speaker SPEAKER_03 (1111.38s - 1112.21s):  A temprano.
Speaker SPEAKER_03 (1113.09s - 1113.39s):  Ah.
Speaker SPEAKER_03 (1114.68s - 1115.69s):  메롱
Speaker SPEAKER_03 (1116.40s - 1118.35s):  I don't know if it's English.
Speaker SPEAKER_01 (1118.35s - 1118.73s):
Speaker SPEAKER_03 (1118.73s - 1120.65s):  Domino effect.
Speaker SPEAKER_01 (1118.94s - 1120.55s):  Domino effect. Domino effect.
Speaker SPEAKER_03 (1120.97s - 1121.98s):  و او فيه
Speaker SPEAKER_03 (1122.57s - 1126.52s):  Kayak talagon lahat illegal terminal setahap.
Speaker SPEAKER_03 (1127.06s - 1127.38s):  You.
Speaker SPEAKER_03 (1128.66s - 1133.20s):  Inya, ina-hami-namo siya. At the least that we can do is to clear this risk.
Speaker SPEAKER_03 (1133.85s - 1134.67s):  the long term.
Speaker SPEAKER_03 (1135.23s - 1138.82s):  We local government should manage.
Speaker SPEAKER_03 (1139.57s - 1142.57s):  We cannot solve even Jaiika USA.
Speaker SPEAKER_03 (1143.21s - 1147.78s):  I don't know if UN has a study on it, but I am Deputy Jai Khan, USAID.
Speaker SPEAKER_03 (1148.24s - 1154.72s):  Asian Development Bank came up with an study. Yes, very long. The traffic cost was about 2.4 billion.
Speaker SPEAKER_02 (1151.82s - 1152.96s):  Yes.
Speaker SPEAKER_00 (1152.96s - 1153.45s):  Thank you.
Speaker SPEAKER_03 (1155.61s - 1163.76s):  and about 2030 it will cost us 30 billion pesos so we really have to address it and I think national government
Speaker SPEAKER_03 (1164.27s - 1166.52s):  with the Build, Build, Build Programme should hurry up.
Speaker SPEAKER_03 (1166.92s - 1169.62s): ityipreää s sırtin nousen täysin sanoon перепakiushinagaima.
Speaker SPEAKER_03 (1169.99s - 1173.70s):  y en quer了吧 muy nueva
Speaker SPEAKER_02 (1172.03s - 1181.58s):  ... okay, you mentioned an interesting point to Nina about Hindi lang local government na patito ono, it's not just the main course, but...
Speaker SPEAKER_03 (1179.76s - 1181.55s):  It's, not just the Navy, but.
Speaker SPEAKER_03 (1181.58s - 1182.14s):
Speaker SPEAKER_02 (1182.14s - 1185.43s):  and the heart everyone has to get involved.
Speaker SPEAKER_03 (1184.25s - 1186.36s):
Speaker SPEAKER_02 (1186.36s - 1186.70s):  Yes.
Speaker SPEAKER_03 (1186.55s - 1189.52s):  Don't get me wrong, I don't want to blame any of you.
Speaker SPEAKER_03 (1189.99s - 1192.10s):  I just am asking you to participate.
Speaker SPEAKER_02 (1192.87s - 1197.97s):  Do you have other ideas also how, or do our folks like us can participate? Yes.
Speaker SPEAKER_01 (1195.76s - 1196.59s):  Or did I folks like
Speaker SPEAKER_03 (1197.53s - 1197.55s):  you
Speaker SPEAKER_01 (1197.55s - 1205.46s):  If I may, that's a very good point. Sure. And the fact that brothers Bredo brought it out. And he mentioned the word I wanted to hear with this citizenship.
Speaker SPEAKER_02 (1198.81s - 1199.24s):  Thank you.
Speaker SPEAKER_01 (1206.85s - 1218.19s):  We all are citizens of this metropolis and of the individual cities. Yet we don't identify ourselves as such. We've lost the connection where we live.
Speaker SPEAKER_00 (1213.17s - 1213.46s):  yet.
Speaker SPEAKER_01 (1218.84s - 1222.07s):  We're more lasalians or I'm sorry, at the end.
Speaker SPEAKER_01 (1222.73s - 1225.31s):  than we are of where we were.
Speaker SPEAKER_01 (1225.97s - 1230.37s):  So the first thing I ask of people is where do you live and what do you call yourselves?        
Speaker SPEAKER_03 (1230.52s - 1235.35s):  k roughly na nilusakwan kami mo 🔔.
Speaker SPEAKER_01 (1233.49s - 1234.23s):  Thank you.
Speaker SPEAKER_01 (1234.83s - 1235.03s):
Speaker SPEAKER_00 (1235.03s - 1235.53s):  Yeah.
Speaker SPEAKER_01 (1235.35s - 1235.37s):  you
Speaker SPEAKER_01 (1235.53s - 1235.62s):
Speaker SPEAKER_03 (1237.34s - 1237.36s):  you
Speaker SPEAKER_01 (1237.36s - 1238.18s):  Ни утра война!
Speaker SPEAKER_01 (1239.45s - 1243.85s):  Pero iyan nga po. Iyong tinata ng pupusam ng audience ko.
Speaker SPEAKER_01 (1244.75s - 1248.85s):  Anong tinata ako nisasarili niyo. If you're from Manile, you're called Manile. Niyo.
Speaker SPEAKER_02 (1248.85s - 1249.32s):  correct.
Speaker SPEAKER_01 (1249.68s - 1250.74s):  ba atang may nila?
Speaker SPEAKER_01 (1251.30s - 1253.42s):  If you're from Cassin City or cold.
Speaker SPEAKER_02 (1254.40s - 1255.56s):  Thank you.
Speaker SPEAKER_01 (1254.96s - 1258.08s):  I used to live in Cassian City, I called myself JEP Prox.
Speaker SPEAKER_00 (1258.53s - 1264.46s):  ja tablespoons,
Speaker SPEAKER_01 (1261.59s - 1276.83s):  But the point is that if we are to build better cities, we have to make reconnect people to where they live. If they don't feel pride in where they live, they won't call themselves off that place. And that's what we lack today.        
Speaker SPEAKER_00 (1270.38s - 1270.74s):
Speaker SPEAKER_02 (1270.74s - 1270.75s):  you
Speaker SPEAKER_02 (1277.37s - 1286.51s):  Parang if you notice also, masyadong segregated, parang yung problema, problema ng manila yan, problema ng Kesun City, problema ng Pasi, problema ng Makati.
Speaker SPEAKER_02 (1287.04s - 1288.50s):  without integrated.
Speaker SPEAKER_02 (1288.81s - 1289.99s):  approach I think.
Speaker SPEAKER_03 (1289.96s - 1290.53s):  No, no.
Speaker SPEAKER_02 (1290.39s - 1291.29s):
Speaker SPEAKER_03 (1291.34s - 1293.03s):  No, no, no, no, that's unfair.
Speaker SPEAKER_02 (1293.03s - 1293.28s):
Speaker SPEAKER_03 (1293.28s - 1295.19s):  because we have
Speaker SPEAKER_03 (1295.49s - 1296.67s):  Metromanila.
Speaker SPEAKER_03 (1297.08s - 1299.03s):  I'm meeting. I'm in the A.
Speaker SPEAKER_02 (1298.21s - 1299.14s):  MNDA.
Speaker SPEAKER_03 (1299.14s - 1299.34s):
Speaker SPEAKER_02 (1299.34s - 1299.36s):  you
Speaker SPEAKER_02 (1299.66s - 1301.18s):  But the problem is...
Speaker SPEAKER_02 (1301.31s - 1312.42s):  And this problem has been there forever. Metro Manila consists of mayors who are elected, 1617 mayors who are elected.
Speaker SPEAKER_02 (1312.97s - 1318.12s):  And you have MMDA that's led by someone who is not elected, but who is appointed.
Speaker SPEAKER_02 (1318.74s - 1326.51s):  ...and ang papel niya is to coordinate. So para ang ako ang mayor, bakit ka magdivicta sa akin, 
ehuala ka naman hina-lal.
Speaker SPEAKER_03 (1327.28s - 1328.41s):  will.
Speaker SPEAKER_02 (1327.81s - 1331.22s):  Isn't that the problem also structurally organizationally?
Speaker SPEAKER_03 (1331.81s - 1333.59s):  Well, that I agree that
Speaker SPEAKER_03 (1334.15s - 1336.75s):  met Romani la chairma should be unelected.
Speaker SPEAKER_03 (1337.27s - 1337.95s):  official
Speaker SPEAKER_03 (1338.59s - 1339.57s):  because
Speaker SPEAKER_00 (1339.57s - 1339.72s):  Thank you.
Speaker SPEAKER_00 (1339.96s - 1340.21s):
Speaker SPEAKER_00 (1340.24s - 1340.28s):  you
Speaker SPEAKER_00 (1340.33s - 1341.75s):  would have solved that, I think, not?
Speaker SPEAKER_00 (1341.96s - 1349.69s):  Federalism. Federalism would have made NCR a region with one elected governor. Tamaba, Yorme.   
Speaker SPEAKER_03 (1350.01s - 1351.20s):  일로, tua고자 hiking!
Speaker SPEAKER_03 (1351.33s - 1359.06s):  I don't want to change government because of problems. I just, you know, there is a little bit, 
okay.
Speaker SPEAKER_00 (1351.43s - 1351.92s):  Aitou?
Speaker SPEAKER_03 (1359.50s - 1364.71s):  There's still marijuana. Actually, in fact, I'll be honest.
Speaker SPEAKER_02 (1361.59s - 1362.13s):  Actually.
Speaker SPEAKER_03 (1365.44s - 1367.38s):  I'll tell you honestly my opinion.
Speaker SPEAKER_03 (1367.78s - 1371.88s):  You can call it communism, you can call it capitalism monarchy.
Speaker SPEAKER_03 (1372.04s - 1377.37s):  Federalism, Parliamentary, and material of forms of government.
Speaker SPEAKER_03 (1379.12s - 1380.22s):  A tyśmy go bierną.
Speaker SPEAKER_03 (1380.42s - 1382.19s):  I'm going to go be out of the toilet, let's work on it.
Speaker SPEAKER_03 (1382.73s - 1385.27s):  In this case, nabang gitlang, yung...
Speaker SPEAKER_03 (1385.79s - 1392.03s):  yung paggiging sebilwala ng mandatuk ng tao, obir sa aming mga...
Speaker SPEAKER_03 (1392.35s - 1399.41s):  Mimang datu ng tal. It makes sense that point. Ya, I agree. We should elect Metromanila Governor.
Speaker SPEAKER_00 (1392.52s - 1393.35s):  Maldito.
Speaker SPEAKER_00 (1394.11s - 1394.33s):
Speaker SPEAKER_03 (1399.85s - 1402.02s):  És un govern, no, regular, govern.
Speaker SPEAKER_03 (1402.07s - 1402.60s):  But...
Speaker SPEAKER_03 (1403.25s - 1404.69s):  setingasain datang
Speaker SPEAKER_03 (1404.98s - 1406.39s):  ja otoreitin.
Speaker SPEAKER_03 (1406.60s - 1411.98s):  kami na man in fairness naman. Anag ako sa pusap naman kami. Kaya lang, there is still this...  
Speaker SPEAKER_03 (1412.42s - 1413.36s):  What we call it.
Speaker SPEAKER_03 (1413.77s - 1414.53s):  guilty
Speaker SPEAKER_03 (1415.34s - 1415.91s):  Come here.
Speaker SPEAKER_03 (1415.98s - 1417.08s): สาธิสัติ สาธิสัติ
Speaker SPEAKER_03 (1417.33s - 1419.07s):  gibrolado, para fracción.
Speaker SPEAKER_03 (1419.54s - 1420.42s):  mentality.
Speaker SPEAKER_03 (1421.11s - 1422.48s):  We still guilty of that.
Speaker SPEAKER_03 (1422.73s - 1424.87s):  Why? We are susceptible.
Speaker SPEAKER_03 (1425.53s - 1428.11s):  addressing our own constituency.
Speaker SPEAKER_03 (1428.80s - 1430.59s):  Now while at the same time,
Speaker SPEAKER_03 (1430.81s - 1431.37s):  Um
Speaker SPEAKER_03 (1431.60s - 1433.29s):  In a way, we try to work.
Speaker SPEAKER_03 (1433.85s - 1434.59s):  한은 안
Speaker SPEAKER_03 (1435.23s - 1448.90s):  to harmonize traffic rules, opening approach. Boehoy Lane is one of the major agreements that we had as Metro Manila Mayor's. Thank you.
Speaker SPEAKER_00 (1439.99s - 1440.46s):
Speaker SPEAKER_00 (1447.74s - 1451.40s):  Minha professor comunquegets na época mainviver grub
Speaker SPEAKER_00 (1451.77s - 1458.77s):  sa kami yung New Yorker. Kayakunapan sinpunin yung sabay sabay nakamin ang sususped ang klasin. 
Speaker SPEAKER_01 (1451.87s - 1451.94s):  Thank you.
Speaker SPEAKER_03 (1451.94s - 1452.04s):  Thank you.
Speaker SPEAKER_02 (1452.04s - 1453.14s):  a million dollars.
Speaker SPEAKER_03 (1457.86s - 1458.60s):  issue long.
Speaker SPEAKER_02 (1458.77s - 1458.94s):
Speaker SPEAKER_03 (1458.94s - 1459.04s):
Speaker SPEAKER_00 (1459.04s - 1459.13s):
Speaker SPEAKER_00 (1460.53s - 1461.03s):  Ha ha ha!
Speaker SPEAKER_02 (1461.96s - 1462.37s):  Thank you.
Speaker SPEAKER_03 (1463.33s - 1463.77s):  See you.
Speaker SPEAKER_03 (1464.38s - 1473.72s):  Umog moong Koga nang. Kekantikkot nagsap nha ng tips by ge o
Speaker SPEAKER_02 (1464.41s - 1464.48s):
Speaker SPEAKER_00 (1464.48s - 1464.54s):
Speaker SPEAKER_02 (1464.54s - 1464.66s):
Speaker SPEAKER_00 (1464.66s - 1464.95s):
Speaker SPEAKER_02 (1464.95s - 1465.05s):
Speaker SPEAKER_01 (1473.89s - 1481.79s):  ...p läng↘ K まamat.
Speaker SPEAKER_03 (1479.92s - 1494.45s):  Oh, I will find more. I will find more. The impairment of the mayor, Abalos, mayor, olive barris, Marikina, everybody is trying to participate and communicate using your technology.
Speaker SPEAKER_03 (1494.73s - 1496.22s):  which is free.
Speaker SPEAKER_03 (1496.52s - 1497.06s):
Speaker SPEAKER_03 (1497.33s - 1504.49s):  Kaya mis sa merukamin, gis mis talagang among ourselves sa ano, ano ng oras mong diniklarak. Sinaka, digitimair is pa ng ala.
Speaker SPEAKER_02 (1502.94s - 1505.79s):  You are going to use your Spanish completely no-we're gonna use it, dude..' okay
Speaker SPEAKER_03 (1505.70s - 1514.19s):  It helps to communicate, it helps to integrate, it helps to reintegrate, it helps to Anishina Binia.
Speaker SPEAKER_03 (1514.29s - 1522.59s):  President Sukhli Dokanina, the collaboration, something to that effect.
Speaker SPEAKER_03 (1522.86s - 1526.81s):  Yes, it helps. Let's see if we get familiar with our problems.
Speaker SPEAKER_03 (1527.12s - 1527.51s):  then
Speaker SPEAKER_03 (1527.79s - 1531.10s):  Going back, doing a traffic parade.
Speaker SPEAKER_03 (1531.54s - 1533.58s):  Y es, oíste el parroquial.
Speaker SPEAKER_03 (1533.93s - 1540.68s):  At some level, but we agreed already with Major Toro Pairs, Mabuhai Lanes.
Speaker SPEAKER_00 (1534.12s - 1534.20s):
Speaker SPEAKER_03 (1540.85s - 1547.40s):  Tapos yung ibang ina-adres na man, ah, our side streets and so on and so forth.
Speaker SPEAKER_02 (1547.55s - 1561.95s):  I'd like to tell you about a survey that Rappler ran among its readers and interesting your results. So, hindi naman ganoong kaisurpising yung results, kasi lumabas dintalaga. We ask our readers about their wish list.
Speaker SPEAKER_02 (1562.40s - 1570.57s):  kung baga, ang gusto ninyon ay prioritais ng mga mayors at least in Metro Manila. And this is what came out.
Speaker SPEAKER_02 (1570.98s - 1571.75s):  Ha ha.
Speaker SPEAKER_02 (1573.00s - 1578.94s):  Congratulations of classes though. Number one is traffic and transportation.
Speaker SPEAKER_01 (1575.87s - 1576.34s):  one.
Speaker SPEAKER_02 (1579.21s - 1601.03s):  47% overwhelmingly, 47% close to half of respondents. Habitila, unahan yung traffic and transportation. So mass transportation, public transportation. Tapo second is lower cost of living. So I guess, nara randa man nara rinto ng tao eh. That it's become so much more expensive and the income is not enough.
Speaker SPEAKER_02 (1601.65s - 1604.84s):  So, me economics, the land.
Speaker SPEAKER_02 (1605.16s - 1605.48s):  end.
Speaker SPEAKER_02 (1605.70s - 1609.16s):  Pero ang layo, compared to traffic, 12.6% lang.
Speaker SPEAKER_02 (1609.57s - 1611.76s):  and then third, this crime prevention.
Speaker SPEAKER_02 (1612.05s - 1619.04s):  Fourth is Health Services Accessibility, and fifth is support for education services.
Speaker SPEAKER_02 (1621.01s - 1623.25s):  صحيحة مع الله طوزوا منك thought
Speaker SPEAKER_03 (1621.06s - 1621.85s):  Good night.
Speaker SPEAKER_00 (1623.36s - 1789.57s):  before, sorry, I'll just mention, so those, I guess not everybody, or not a lot live in Keson City here, but for those of you who do, since you're talking about traffic, I'd like to wind them ahead of time, no, that it will get worse before it will get better. Why is that? Because four of the main build, build, build projects of our president are concentrated in Keson City. For example, 12 of the 14 MRT stations are in Keson City. The seven of the 15 subway stations are in Keson City. The segment 8.2, which 
is the connector road, connect pin C5 to NLX, it will be in Keson City. And the skyway connecting S-Lex and NLX is in Keson City, passes through Keson City as well. And because the president is rushing, he would like these build, build, build projects to finish before his term 
ends, which means for us in Keson City, at least and possibly also in Manila, that all of these construction projects will be taking place 
at the same time. And so in our case, having a had that foresight, that much, that's a problem in traffic, much, much grabbersha and we've 
talked to the DOTR, DPWH, they said, we don't have time to do it, because it's a plan of the past. We set up a task force, it's called task force, build, build, address, build, build, build. So meaning not only that because when the connector road is built, 18,000 of my informal settler families will be affected. And I have to build housing for them in relation to this. So it's actually quite a difficult situation. But that's why I guess I agree with Mayor Eskot, that the people must participate, they must know that this is happening and the government is now in coordination with various sectors in the city, informing them that this will happen inevitably. What are the mitigating measures that we can put in place now, palang, habang, pinaplan, ito national government so that we can anticipate the inconvenience it will cause and we can adjust our lifestyles while this is going to happen. And then hopefully if the national government fulfills its promise that within X number of years these building or construction projects will be finished, the promise is of course public transportation, which will then ease or lower the cost of living and the traffic situation that you mentioned are at the top of the list of our people. But I'd like to emphasize before there is game, there is pain. And I guess we are entering the pain phase.
Speaker SPEAKER_02 (1673.19s - 1673.91s):  and be gusto.
Speaker SPEAKER_01 (1673.91s - 1673.96s):
Speaker SPEAKER_02 (1790.10s - 1807.06s):  Okay, so you have to be ready for this. On average, we're told that Filipinos now spend about 66 minutes on the road because of traffic. Daily, daily. So if you're saying that things will get worse, in 66 minutes will probably become double.
Speaker SPEAKER_02 (1808.19s - 1810.26s):  How can we alleviate this?
Speaker SPEAKER_02 (1810.75s - 1817.00s):  Magiging masmanageable ito. If it's going to get worse anyway, and we can't do anything about it because of the infrastructure.
Speaker SPEAKER_03 (1818.53s - 1819.43s):  Well, uh...
Speaker SPEAKER_03 (1820.17s - 1821.62s):  Acceptance of facts.
Speaker SPEAKER_03 (1824.96s - 1826.41s):  That's to start with.
Speaker SPEAKER_03 (1827.59s - 1828.25s):  Ok, sí.
Speaker SPEAKER_03 (1828.98s - 1832.37s):  Inangan o, halimbawa. Limbawa lang. Let's just say...
Speaker SPEAKER_03 (1832.81s - 1834.73s):  all these infrastructures.
Speaker SPEAKER_03 (1835.15s - 1836.84s):  or infrastructure is true.
Speaker SPEAKER_03 (1838.17s - 1838.98s):  Let's just see.
Speaker SPEAKER_03 (1839.34s - 1842.38s):  all of this will happen. Let's just say, let's just continue to hope.
Speaker SPEAKER_03 (1843.05s - 1845.38s):  So what can we do?
Speaker SPEAKER_03 (1846.43s - 1847.22s):  as I sit this end.
Speaker SPEAKER_03 (1848.27s - 1849.40s):  إحدكتي movie
Speaker SPEAKER_03 (1850.39s - 1851.15s):  serílico.
Speaker SPEAKER_03 (1851.59s - 1852.87s): ien – 그 oyster Jackson
Speaker SPEAKER_03 (1853.83s - 1871.03s):  that if I'm suffering today about, let me borrow that data, 66 minutes of my life from point A to point B, then with these programs, infrastructure programs,
Speaker SPEAKER_03 (1871.40s - 1872.36s):  Συνάβαιε!
Speaker SPEAKER_03 (1872.45s - 1875.50s):  It takes about 30 minutes so that's going to be...
Speaker SPEAKER_03 (1875.69s - 1876.72s):  and I will
Speaker SPEAKER_03 (1877.24s - 1878.66s):  and 36 minutes.
Speaker SPEAKER_03 (1879.25s - 1881.21s):  So, me and your baby will go into your lifestyle.
Speaker SPEAKER_03 (1882.18s - 1882.88s):  I will let that.
Speaker SPEAKER_03 (1883.79s - 1884.53s):  As a citizen.
Speaker SPEAKER_03 (1884.88s - 1886.74s):  that is my share to this state.
Speaker SPEAKER_03 (1886.98s - 1890.25s):  That is my share to the government. I'll pay the price.
Speaker SPEAKER_03 (1890.79s - 1894.60s):  and I hope this government officials will be true to their words.
Speaker SPEAKER_03 (1895.25s - 1897.51s):  Конgartνο Finnish όσοι τις Μιλσκουχμερές.
Speaker SPEAKER_02 (1899.16s - 1901.25s):  promise.
Speaker SPEAKER_03 (1899.85s - 1903.68s):  Okay, I mean, you're in support of the Dayong Mamma Mayan.
Speaker SPEAKER_02 (1903.68s - 1904.05s):  time can see.
Speaker SPEAKER_03 (1903.94s - 1904.83s): ก็มีน้ำห้าน
Speaker SPEAKER_03 (1905.03s - 1907.14s):  bila ng taong gubiarno.
Speaker SPEAKER_03 (1907.58s - 1908.07s):  Uh.
Speaker SPEAKER_03 (1908.75s - 1922.92s):  The least thing that we can do for you, we as local government, I cannot speak in behalf of the 
National Government. The least thing that we can do for you, like in the case of Manila, if you're traveling from point A to point B at two hours a day, if I can give you one forty-five.
Speaker SPEAKER_03 (1924.14s - 1929.18s):  That's what you call management. If I can do better, like one hour and 30 minutes,
Speaker SPEAKER_03 (1929.70s - 1930.56s):  that is the
Speaker SPEAKER_03 (1931.31s - 1933.47s):  Second to better is what?
Speaker SPEAKER_03 (1934.50s - 1936.67s):  wang gumang nangkonte.
Speaker SPEAKER_03 (1937.16s - 1941.58s):  Tapi aparat itu grouped di KUNA été bah pleasure. Dus, Brisbanenya akan åker Indian jauh adalah 
lebihpps,
Speaker SPEAKER_00 (1939.88s - 1940.03s):
Speaker SPEAKER_02 (1940.03s - 1940.40s):  Thank you.
Speaker SPEAKER_00 (1940.40s - 1940.45s):
Speaker SPEAKER_03 (1942.07s - 1943.42s):  Simple Science.
Speaker SPEAKER_03 (1944.39s - 1945.52s):  Lesser space.
Speaker SPEAKER_03 (1945.85s - 1947.10s):  Ok, bye bye, mate.
Speaker SPEAKER_03 (1948.25s - 1949.30s):  It's lesser space.
Speaker SPEAKER_03 (1950.85s - 1957.48s):  In the case of Manila, we have a blessed space, basically, because it's about almost 500-year-old city.
Speaker SPEAKER_02 (1951.07s - 1951.93s):  kay iso commitment<|tl|>
Speaker SPEAKER_03 (1957.89s - 1963.15s):  much smaller than Cassian City. And it was designed by our ancestors.
Speaker SPEAKER_02 (1958.02s - 1960.01s):  much smaller than Cassian city.
Speaker SPEAKER_03 (1963.30s - 1965.38s):  Naturине,رجatkan temúsuan pun. Aku preservekan impressive dalam teman. tuduh. Apa pun, guitarfork<|fo|>u saya akan memunyiakantersikan lebih memuasak. Tetapi jadi, duit untuk sejak untuk2]. Ibu kita boleh hatikan sejak itu. B sava,    
Speaker SPEAKER_03 (1965.72s - 1968.15s):  Maybe at the time, I don't know with their mathematics.
Speaker SPEAKER_03 (1968.48s - 1970.91s):  untuk adaptuh 100.000 populasi.
Speaker SPEAKER_03 (1971.52s - 1972.52s):  Manila is about to...
Speaker SPEAKER_03 (1972.84s - 1975.00s):  500 years after
Speaker SPEAKER_03 (1975.42s - 1979.59s):  It's about 1.89 population and 3 million data in population.
Speaker SPEAKER_03 (1979.81s - 1984.58s):  and Manila is the host of all universities except for Atteneo.
Speaker SPEAKER_03 (1984.80s - 1985.73s):
Speaker SPEAKER_01 (1986.37s - 1988.56s):  Get in, in you piece, in you piece.
Speaker SPEAKER_03 (1988.46s - 1990.69s):  And you pee, na may you pee manila kami.
Speaker SPEAKER_03 (1990.93s - 1994.12s):  So we host a lot of students.
Speaker SPEAKER_03 (1994.35s - 1996.06s):  That's why
Speaker SPEAKER_03 (1996.43s - 1996.78s):  Uh,
Speaker SPEAKER_03 (1997.29s - 2000.82s):  the infernility to the students, they really...
Speaker SPEAKER_03 (2001.07s - 2006.13s):  Hang on to the decision of the Mayor of Manila with regard to classes.
Speaker SPEAKER_03 (2006.38s - 2011.01s):  because they want to suspend classes because they know.
Speaker SPEAKER_03 (2011.33s - 2012.90s):  the experience plugged in.
Speaker SPEAKER_03 (2013.35s - 2014.72s):  the experience traffic.
Speaker SPEAKER_03 (2015.09s - 2016.95s):  experience chaos.
Speaker SPEAKER_02 (2016.09s - 2017.91s):  But my question is...
Speaker SPEAKER_02 (2018.30s - 2019.51s):  Ganan alang bayon.
Speaker SPEAKER_02 (2019.65s - 2023.83s):  Iti iti isin ng naten, and we adopt, and be resilient.
Speaker SPEAKER_02 (2024.24s - 2026.31s):  That may be, uh, again.
Speaker SPEAKER_02 (2026.70s - 2029.32s):  just one one this one from all over the world.
Speaker SPEAKER_01 (2027.41s - 2033.55s):  I think everyone will agree that they will, if you want to ease the pain.
Speaker SPEAKER_01 (2034.46s - 2036.37s):  if you're really expected to
Speaker SPEAKER_01 (2036.64s - 2040.39s):  double our commute time is that they
Speaker SPEAKER_01 (2040.66s - 2043.19s):  make our Wi-Fi twice as fast.
Speaker SPEAKER_01 (2043.76s - 2046.11s):  but at one half the cost, right?
Speaker SPEAKER_02 (2046.60s - 2048.94s):  So you can work anywhere.
Speaker SPEAKER_01 (2049.18s - 2052.05s):  You can browse. And then we will suffer in silence.
Speaker SPEAKER_02 (2049.74s - 2049.89s):  Thank you.
Speaker SPEAKER_02 (2052.89s - 2052.94s):
Speaker SPEAKER_02 (2052.98s - 2053.28s):
Speaker SPEAKER_03 (2053.28s - 2053.79s):  Okay.
Speaker SPEAKER_03 (2054.49s - 2056.50s):  No, that thing of...
Speaker SPEAKER_03 (2057.50s - 2060.60s):  My suggestion is just my suggestion.
Speaker SPEAKER_03 (2060.92s - 2063.40s):  It doesn't mean that you have to do it, it's up to you.
Speaker SPEAKER_03 (2063.73s - 2067.24s):  Karena jika kamu ingin memiliki hidup, jangan menghubungi hidup.
Speaker SPEAKER_03 (2067.71s - 2068.94s):  We were not going to accept The Fox first.
Speaker SPEAKER_03 (2069.04s - 2069.61s):  Lenny.
Speaker SPEAKER_03 (2069.77s - 2070.83s):  talın ciinen janganin
Speaker SPEAKER_03 (2071.27s - 2072.58s):  يجب أن闖ل هناك تurfسين
Speaker SPEAKER_03 (2073.29s - 2075.77s):  KUPS ng mga nasagobiar nuyo.
Speaker SPEAKER_03 (2076.13s - 2076.58s):  language
Speaker SPEAKER_03 (2078.31s - 2079.59s):  but Si Gai!
Speaker SPEAKER_03 (2080.01s - 2081.14s):  Once and for all.
Speaker SPEAKER_03 (2081.56s - 2083.55s):  once in for all.
Speaker SPEAKER_03 (2084.45s - 2086.84s):  ‫אז אתם יכולים לך, ‫אז אתם יכולים לך.
Speaker SPEAKER_03 (2088.08s - 2089.51s):  Maybe he's a serious guy.
Speaker SPEAKER_03 (2090.15s - 2092.06s):  Maybe it's really gonna do something.
Speaker SPEAKER_03 (2092.40s - 2093.70s):  ... about the situation.
Speaker SPEAKER_03 (2094.24s - 2097.15s):  because if you do that, now you bring back hope.
Speaker SPEAKER_03 (2097.73s - 2100.70s):  to your government. You're trying to revive.
Speaker SPEAKER_03 (2101.14s - 2102.44s):  Hope to your side.
Speaker SPEAKER_03 (2102.77s - 2106.27s):  Awak mengambakkan bahagian serius setunggu biar itu. Tapi kami namaan...
Speaker SPEAKER_03 (2106.33s - 2114.06s):  Yung nga yung pangaku nami senyo, yung compromiso nami senyo. Pipiliitinamen sa abot nga ming mahal kaya yung pede namin ki participi.
Speaker SPEAKER_03 (2114.84s - 2119.29s):  In fact, Argentina should ask him that
Speaker SPEAKER_03 (2119.80s - 2124.61s):  There are things that you don't know that we are planning and trying to... Could you tell us?   
Speaker SPEAKER_02 (2123.88s - 2127.73s):  Could you tell us about that? Maybe because we have limited time.
Speaker SPEAKER_03 (2127.12s - 2128.04s):  Yeah, Greg!
Speaker SPEAKER_02 (2128.04s - 2136.05s):  The last question is from the three of you. Maybe share your plans or your insights about it.   
Speaker SPEAKER_03 (2128.10s - 2128.37s):  Thank you.
Speaker SPEAKER_02 (2136.32s - 2143.14s):  how things can be better and how living in the city in the metro can be a better experience.    
Speaker SPEAKER_00 (2145.00s - 2155.07s):  Well, in our case, what we're trying to do is build several growth centers in Kessend City. So Kessend City is probably the opposite of the menu. We have
Speaker SPEAKER_00 (2155.39s - 2305.75s):  Manila has about half the population of Cassant City, but we are about five times bigger in terms of land area. So we are able to plan for several growth areas. So it's not necessary for us to have just one city center. We can have many city centers. For example, the Kubau can be a city center, the central business district, which is the Trinoma area, can be a central center. The Novaleaches area can be a center. Balintawa can be a center, et cetera. So that's how we're planning to organize our city in terms of different growth areas. So that people do not need to travel long distances to get to places of work. At the same time, you make each area attractive to investors so that each can be its own microcosm. Completo nasanan jane yumangan negociante, nana completona ambawath growth area. So people, this will minimize transportation needs. In other words, I think our country patterned itself too much on the American 
model, which is car-based, which is distance oriented suburban in lifestyle, but now we have to try to adopt the European model, which is compact city centers in our spaces, not to maximize walkability, pedestrianization, and mixed use of facilities. And I guess that's the way 
our city is trying to organize itself. Now, of course, you're asking in the short term, I'm kind of a long term thinker. That's why probably I hope that I make it to the end of my nine years. Bah parea, taya, mayor, escoria, gambling, or political capital, because there's so many things we want to do. But actually, in the long term, we have many nice plans for our city. But in the short term, I'll just very briefly say what we're trying to do now is decentralize governance in Guestan City. So all of the malls in Guestan City will now, by January, we 
have one stop shops where people can pay their taxes. And they don't have to go to the monocity, they don't have to line up anymore. We're 
trying to move towards a fully automated city by the end of three years. But at least the revenue generating part by the end of this year should be finished. And then we go on and on until at the end of three years, hopefully everything will be online. And people no longer have to go to the city government, the city hall to get transactions done. That's our plan for our city.
Speaker SPEAKER_01 (2252.05s - 2252.79s):  That's funny.
Speaker SPEAKER_02 (2305.12s - 2310.93s):  Okay, so for those, I don't know if there are any of you from Casson City here that's something 
to look forward to.
Speaker SPEAKER_02 (2311.43s - 2311.84s):  봐봐요.
Speaker SPEAKER_01 (2311.84s - 2319.20s):  I think after there are nine years, the next step would be for everyone to vote for either of them as governor of Metro Manila.
Speaker SPEAKER_01 (2320.97s - 2327.63s):  But just one very quick, and this is very good to have such a discourse on cities and quality of our lives here.
Speaker SPEAKER_01 (2327.79s - 2334.94s):  I would invite you all for those of you who haven't finished your degrees yet. To think about the career in urban planning.
Speaker SPEAKER_01 (2335.36s - 2358.03s):  They're only 6,000 of us in the profession. There are 60,000 architects. We have world-class buildings, but once you step out, you get run over. So we need, and all of this mayors, their plantilias require urban planners and landscape 
architects and urban designers. We need people in the planning professions to be able to help us all.
Speaker SPEAKER_01 (2358.46s - 2360.29s):  move to that brighter future.
Speaker SPEAKER_02 (2361.86s - 2362.41s):  Thank you.
Speaker SPEAKER_02 (2363.22s - 2363.73s):  Scott.
Speaker SPEAKER_03 (2364.29s - 2366.43s):  anong letaw na nakalimuot ang con音
Speaker SPEAKER_02 (2366.43s - 2366.77s):
Speaker SPEAKER_02 (2367.96s - 2369.75s):  So I knew...
Speaker SPEAKER_02 (2370.02s - 2370.09s):  you
Speaker SPEAKER_03 (2370.09s - 2371.90s):  No, I'm just kidding.
Speaker SPEAKER_02 (2370.92s - 2371.86s):  I'm just kidding.
Speaker SPEAKER_02 (2371.90s - 2372.25s):  lol
Speaker SPEAKER_03 (2375.96s - 2378.75s):  If you know the principle of ground survival.
Speaker SPEAKER_03 (2380.69s - 2382.80s):  I would rather invest in that.
Speaker SPEAKER_03 (2383.66s - 2384.57s):  Pram de baotam ha.
Speaker SPEAKER_03 (2385.75s - 2386.86s):  Go back to Bay City.
Speaker SPEAKER_03 (2388.48s - 2393.94s):  You may have all the high paluting.
Speaker SPEAKER_03 (2395.02s - 2397.02s):  Ideas, words.
Speaker SPEAKER_03 (2398.22s - 2399.98s):  deyin ko lang ah, deyin ko lang.
Speaker SPEAKER_03 (2401.07s - 2403.33s):  Kayang hindi na tamipinag akatiwala anong.
Speaker SPEAKER_03 (2404.79s - 2406.64s):  que se vio a ir a pulo promises.
Speaker SPEAKER_03 (2409.29s - 2417.09s):  So, what I am saying is that I think I would rather invest in going back to basic by talking to 
people.
Speaker SPEAKER_03 (2417.48s - 2419.91s):  Listening to people, addressing their...
Speaker SPEAKER_03 (2421.02s - 2424.87s):  recomm da jedaname boli imegad.
Speaker SPEAKER_03 (2425.73s - 2427.99s):  و commit م dokزنا قاتل بها في العمل
Speaker SPEAKER_03 (2428.97s - 2432.53s):  in the long term, because I can always ask technocrats
Speaker SPEAKER_03 (2432.70s - 2433.59s):  on my left.
Speaker SPEAKER_03 (2434.06s - 2434.79s):  to plan.
Speaker SPEAKER_03 (2435.14s - 2440.32s):  while at the same time I address the day-to-day concern of our people.
Speaker SPEAKER_03 (2441.29s - 2446.16s):  ako kasi na niwa lago, to change these things, ke lang ng natao mo'lpartisipit.
Speaker SPEAKER_03 (2446.21s - 2448.36s):  i com a participar és...
Speaker SPEAKER_03 (2448.51s - 2450.72s):  on how are you going?
Speaker SPEAKER_03 (2451.41s - 2454.80s):  to convince others to change their mindset.
Speaker SPEAKER_03 (2457.06s - 2457.82s):  고마워요.
Speaker SPEAKER_03 (2458.20s - 2460.74s):  Medyo ito toon ulang.
Speaker SPEAKER_03 (2461.55s - 2462.01s): 歌詞
Speaker SPEAKER_03 (2462.68s - 2463.78s):  infrastructure.
Speaker SPEAKER_03 (2464.47s - 2469.91s):  sa manewala kais sa Indi, kandikol ako mein aga-arag dito sa gubiyarno. Kaya chong ko ilagaini niyo.
Speaker SPEAKER_03 (2470.28s - 2470.95s):  نميور
Speaker SPEAKER_03 (2472.02s - 2475.73s):  There is what you call appropriation for infrastructure projects.
Speaker SPEAKER_03 (2476.12s - 2478.39s):  h underlands injury.
Speaker SPEAKER_03 (2480.47s - 2481.60s):  So it'd be simply him.
Speaker SPEAKER_03 (2481.65s - 2483.85s):  tentangvoorbeeldan
Speaker SPEAKER_03 (2484.44s - 2485.60s):  with or without
Speaker SPEAKER_03 (2485.97s - 2486.49s):  the mayor.
Speaker SPEAKER_03 (2487.49s - 2489.60s):  because it's designed.
Speaker SPEAKER_03 (2490.02s - 2490.58s):  That way.
Speaker SPEAKER_03 (2491.73s - 2493.30s):  There are things like already...
Speaker SPEAKER_03 (2493.68s - 2494.41s):  in place.
Speaker SPEAKER_03 (2495.42s - 2496.96s):  That's why they be like people.
Speaker SPEAKER_03 (2497.40s - 2498.91s):  in their respective offices.
Speaker SPEAKER_03 (2499.49s - 2500.08s):  to guide.
Speaker SPEAKER_03 (2501.70s - 2503.57s):  great policy and visions.
Speaker SPEAKER_03 (2504.26s - 2506.04s):  So my greatest dream.
Speaker SPEAKER_03 (2507.20s - 2508.80s):  Maaf mana мereka sangat sempur.
Speaker SPEAKER_03 (2509.83s - 2510.68s):  una gente que se dice amplio. Pero en la internetыватьás 직 posiblados les ha escuchado un poco 
para bleeding. Y... childcare.... el país vida felices no se fibre su casa. ¡Era insulta! Es un autótipoники trains. Unda de jumpos. ¡Año padre no hay contactinités! Después de subirse. ¡No vamos a disparar para a ayudarme! Luke sea mucho ganando. ¡Para la studia solo<|es|> museums que los Drumwittles tienen un gramatón! ¡Nag divorce mariana!!! ¡R fine!.. Para reезar su γ creo que es porque los cuartes tienenructor y poderes sus Его spiccidos para hacer las cosas que se sacan. Para mis niego y hehe,
Speaker SPEAKER_03 (2511.30s - 2514.05s):  Darating naman talaga yan mga pagawain baya.
Speaker SPEAKER_03 (2514.39s - 2515.10s):  kira tingen.
Speaker SPEAKER_03 (2515.98s - 2516.79s):  Thank you.
Speaker SPEAKER_03 (2517.34s - 2518.62s):  More than anything else.
Speaker SPEAKER_03 (2519.08s - 2520.06s):  I want to create
Speaker SPEAKER_03 (2521.02s - 2521.29s):
Speaker SPEAKER_03 (2522.00s - 2523.60s):  a city like an orchestra.
Speaker SPEAKER_03 (2526.42s - 2528.95s):  kompo sa different musical instruments.
Speaker SPEAKER_03 (2531.82s - 2533.12s):  play one music.
Speaker SPEAKER_03 (2534.66s - 2535.38s):  in harmony.
Speaker SPEAKER_03 (2538.10s - 2538.98s):  that all
Speaker SPEAKER_03 (2539.85s - 2540.76s):  sectors.
Speaker SPEAKER_03 (2541.73s - 2542.45s):  issues.
Speaker SPEAKER_03 (2542.98s - 2544.11s):  are being addressed.
Speaker SPEAKER_03 (2544.88s - 2546.40s):  consistently.
Speaker SPEAKER_03 (2547.75s - 2549.03s):  persistently.
Speaker SPEAKER_03 (2551.04s - 2553.22s):  being conducted by one person.
Speaker SPEAKER_03 (2553.62s - 2555.99s):  και τότε αυθώσεις σε έναν κερόμα του Έλλη.
Speaker SPEAKER_00 (2553.91s - 2554.15s):  Thank you.
Speaker SPEAKER_03 (2556.75s - 2558.26s):  being conducted by
Speaker SPEAKER_03 (2558.55s - 2559.16s):  Low.
Speaker SPEAKER_03 (2560.04s - 2560.61s):  in order.
Speaker SPEAKER_03 (2561.84s - 2562.48s):  without it.
Speaker SPEAKER_03 (2563.01s - 2564.02s):  It's nothing to follow.
Speaker SPEAKER_03 (2565.47s - 2567.66s):  So bakal lang nakaligdaan natin.
Speaker SPEAKER_03 (2568.61s - 2569.08s):  Nah.
Speaker SPEAKER_03 (2570.36s - 2570.94s):  Maybe.
Speaker SPEAKER_03 (2571.78s - 2572.27s):  and
Speaker SPEAKER_03 (2572.61s - 2574.58s):  மாறை தலகம்
Speaker SPEAKER_03 (2574.84s - 2576.66s):  Mec Muna Paco Koola ngubierno
Speaker SPEAKER_03 (2576.96s - 2577.43s):  better.
Speaker SPEAKER_03 (2578.53s - 2579.51s):  Can we just...
Speaker SPEAKER_03 (2580.08s - 2581.16s):  sit for a moment.
Speaker SPEAKER_03 (2582.40s - 2584.00s):  just for five minutes.
Speaker SPEAKER_03 (2584.50s - 2585.30s):  Thank you to Over.
Speaker SPEAKER_03 (2587.22s - 2588.25s):  아노만의 몸바.
Speaker SPEAKER_03 (2589.21s - 2590.97s):  まpuedinguma fica en participación.
Speaker SPEAKER_03 (2591.58s - 2592.18s):  So.
Speaker SPEAKER_03 (2593.40s - 2594.04s):  بمال
Speaker SPEAKER_03 (2595.42s - 2598.61s):  That's my dream of Manila.
Speaker SPEAKER_02 (2599.51s - 2602.83s):  Ok, então é claro que é muito participatory, não?
Speaker SPEAKER_03 (2599.71s - 2599.91s):
Speaker SPEAKER_02 (2604.20s - 2604.69s):  Yes.
Speaker SPEAKER_02 (2605.29s - 2612.55s):  So, okay, we don't have enough time anymore, so we'd like to thank...
Speaker SPEAKER_02 (2612.67s - 2627.59s):  our mayors for joining us. I know this is a very complicated and complex problem. Hindi to, I'm 
not going to use that in just 30 minutes, but as they have pointed out, it's really important for each and every one of us to be part of the process.
Speaker SPEAKER_02 (2627.74s - 2641.05s):  and to also hold them into account if they don't deliver on their promises. But bantayan natin ang Isatisa, but citizens have to do their part as well. Thank you very much. Thank you. It's God. Thank you, Joy. And thank you, Paola.
"""

# Clean the transcript and summarize
cleaned_transcript = clean_text(text)
summary = summarize(cleaned_transcript)
print(summary)