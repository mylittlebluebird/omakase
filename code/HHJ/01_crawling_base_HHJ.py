# end of memory error
# end of memory error
# end of memory error
# end of memory error

#선언
import time
import re
import pandas as pd
import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException

#변수 선언
options = ChromeOptions()
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
options.add_argument('user_agent=' + user_agent)
options.add_argument('lang=ko_KR')
options.add_argument('--blink-setting=imagesEnable=false') #이미지 미로딩
#options.add_argument('headless') #화면 제거

#함수 선언
def restaurant_page_down():
    while(1):
        first_restaurant_list = driver.find_elements(By.CLASS_NAME, 'UEzoS')
        action.move_to_element(first_restaurant_list[-1]).perform()
        time.sleep(0.8)
        last_restaurant_list = driver.find_elements(By.CLASS_NAME, 'UEzoS')
        if first_restaurant_list == last_restaurant_list:
            print('Page Down END:', len(first_restaurant_list), len(last_restaurant_list))
            return last_restaurant_list
        else:
            print('Page Down:', len(first_restaurant_list), len(last_restaurant_list))
            continue
def restaurant_next_page():
    btn = driver.find_elements(By.CLASS_NAME, 'eUTV2')
    print(btn[1].get_attribute('aria-disabled'))
    if btn[1].get_attribute('aria-disabled') == 'false':
        btn[1].click()
        return 0
    else:
        print('Last Page')
        return 1

#전체시간
all_time = 0

#리스트
locations = ['교하동', '금촌동', '동패동', '문발동', '서패동', '야당동']

#검색
for location in locations:
    #선언
    service = ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    action = ActionChains(driver)
    url = 'https://map.naver.com/p/search/파주 {} 식당'.format(location)
    driver.get(url)
    time.sleep(5)
    pf = pd.DataFrame()
    names = []
    reviews = []

    # 프레임 변경
    driver.switch_to.default_content()  # 프레임 초기화
    driver.switch_to.frame('searchIframe')  # 프레임 변경

    while(1):
        restaurant_page_down()
        restaurant_lists = driver.find_elements(By.CLASS_NAME, 'TYaxT')

        for restaurant_list in restaurant_lists:
            #시간측정
            start_time = time.time()

            # 몇 번째 처리하는지 출력
            page = driver.find_element(By.CLASS_NAME, 'mBN2s.qxokY').text
            list_num = restaurant_lists.index(restaurant_list)
            print('---------------------------------------------------')
            print('Crawling Start at {}:page, {}:restaurant'.format(page, list_num+1))

            #선언
            review = ''

            restaurant_list.click() #식당 클릭
            names.append(restaurant_list.text) #식당 이름 추가
            time.sleep(3)

            #프레임 변경
            driver.switch_to.default_content()  # 프레임 초기화
            driver.switch_to.frame('entryIframe')  # 프레임 변경

            #리뷰 보기 버튼
            btn_lists = driver.find_elements(By.CLASS_NAME, 'veBoZ')
            for btn_list in btn_lists:
                if btn_list.text == '리뷰':
                    btn_list.click()
            time.sleep(5)

            review_count_all = driver.find_element(By.CLASS_NAME, 'place_section_count')
            count_all = int(re.compile('[^0-9]').sub('', review_count_all.text))
            # 리뷰 더보기 버튼
            while(1):
                review_count = len(driver.find_elements(By.CLASS_NAME, 'zPfVt'))
                try:
                    print('Review Crawling Loading: [',
                          int((review_count/count_all)*100), '% ] [',
                          review_count, '/', count_all, ']')
                    driver.find_element(By.CLASS_NAME, 'E4qxG').click()
                    time.sleep(1)
                except NoSuchElementException:
                    print('Reviews End Point')
                    break
                except:
                    print('Reviews More BTN Error')
                if review_count > 50:
                    break

            # 리뷰 출력
            review_lists = driver.find_elements(By.CLASS_NAME, 'zPfVt')
            for review_list in review_lists:
                try:
                    review = review + ' ' + re.compile('[^가-힣]').sub(' ', review_list.text)
                except:
                    pass
            print('Review Crawling End: ', len(review))
            reviews.append(review)

            # 프레임 변경
            driver.switch_to.default_content()  # 프레임 초기화
            driver.switch_to.frame('searchIframe')  # 프레임 변경

            driver.delete_all_cookies()

            end_time = time.time()
            start_to_end_time = end_time - start_time
            all_time = all_time + int(start_to_end_time)
            print(f"Crawling Running Time: {start_to_end_time:.2f} sec")

        if restaurant_next_page() ==  1:
            break

    time.sleep(1)

    driver.close()
    time.sleep(2)
    driver.quit()
    time.sleep(2)

    pf['names'] = names
    pf['reviews'] = reviews
    pf.to_csv('../data_naver/{}_naver_data_{}_HHJ.csv'.format(location,datetime.datetime.now().strftime('%Y%m%d')),
              index=False)

#전체시간 출력
print(f"Crawling Running Time: {all_time:.2f} sec")
