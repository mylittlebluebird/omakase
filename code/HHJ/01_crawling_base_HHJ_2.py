#선언
import time
import re
import pandas as pd
import glob

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException

#함수선언
def set_restaurant_page(num, page):
    selected_page_num = int(driver.find_element(By.CLASS_NAME, 'mBN2s.qxokY').text)
    print('★PAGE SETTING START★')
    while(selected_page_num != num):
        print('PAGE SETTING    :', selected_page_num, '/', num)
        driver.find_elements(By.CLASS_NAME, 'eUTV2')[1].click()
        time.sleep(0.2)
        selected_page_num = int(driver.find_element(By.CLASS_NAME, 'mBN2s.qxokY').text)
    print('PAGE SETTING END:', selected_page_num, '/', num)
    scroll_down_restaurant(page)

    return

def scroll_down_restaurant(num):
    print('\n★SCROLL DOWN START★')
    time.sleep(1)
    cnt = 0
    while (1):
        contents = driver.find_element(By.CLASS_NAME, 'Ryr1F')
        restaurant_list = contents.find_elements(By.CLASS_NAME, 'UEzoS.rTjJo')
        action.move_to_element(restaurant_list[-1]).perform()
        time.sleep(1)
        if len(restaurant_list) >= num:
            print('\rPage Down END:', cnt, '>', len(restaurant_list), end="")
            break
        else:
            print('\rPage Down    :', cnt, '>', len(restaurant_list), end="")
            cnt = len(restaurant_list)
            continue

    return

#검색어
locations = ['김포 한강신도시']

while(1):
    try:
        #변수선언
        location_num = 0
        page_num = 1
        restaurant_num = 0

        #저장된 데이터 선별 데이터 불러오기
        data_paths = glob.glob('../data_naver/naver*HHJ.csv')
        print(data_paths)

        #저장된 데이터 선별
        for location in locations:
            for i in range(7):
                for j in range(50):
                    if '../data_naver\\naver_data_{}_{}_{}_HHJ.csv'.format(location, i, j) in data_paths:
                        location_num = locations.index(location)
                        page_num = i
                        restaurant_num = j+1
        print('▶Last Save Data = {} / {} / {}'.format(location_num, page_num, restaurant_num))

        #크롤링 변수 선언
        options = ChromeOptions()
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        options.add_argument('user_agent=' + user_agent)
        options.add_argument('lang=ko_KR')
        options.add_argument('--blink-setting=imagesEnable=false') #이미지 미로딩

        #크롤링 시작
        for location in locations[location_num:]:
            for i in range(page_num, 7):
                for j in range(restaurant_num, 50):
                    print('\n\n----------------------------------------')
                    print(' ▶ Target = {} / {} / {}'.format(location, i, j))
                    print('----------------------------------------')
                    # 식당별 저장 변수
                    df = pd.DataFrame()
                    names = []
                    reviews = []

                    # 선언
                    service = ChromeService(executable_path=ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=options)
                    action = ActionChains(driver)
                    url = 'https://map.naver.com/p/search/{} 식당'.format(location)

                    # 웹페이지 OPEN
                    driver.get(url)
                    time.sleep(5)

                    # 프레임 변경
                    driver.switch_to.default_content()  # 프레임 초기화
                    driver.switch_to.frame('searchIframe')  # 프레임 변경
                    time.sleep(0.5)

                    #페이지 설정
                    set_restaurant_page(i, j)

                    #타겟 엘리멘트 설정
                    target = driver.find_elements(By.CLASS_NAME, 'TYaxT')[j]
                    names.append(target.text)
                    time.sleep(0.5)

                    #타겟 클릭
                    print('\n\n★Target Click★')
                    try:
                        target.click()
                        print('Clicked Success')
                    except Exception as e:
                        print('Error Code:', e)

                    time.sleep(3)

                    # 프레임 변경
                    driver.switch_to.default_content()  # 프레임 초기화
                    driver.switch_to.frame('entryIframe')  # 프레임 변경
                    time.sleep(0.5)

                    # 리뷰 보기 버튼
                    print('\n★Review More Click★')
                    btn_lists = driver.find_elements(By.CLASS_NAME, 'veBoZ')
                    for btn_list in btn_lists:
                        if btn_list.text == '리뷰':
                            btn_list.click()
                    time.sleep(2)

                    review_count_all = driver.find_element(By.CLASS_NAME, 'place_section_count')
                    count_all = int(re.compile('[^0-9]').sub('', review_count_all.text))

                    # 리뷰 더보기 버튼
                    while (1):
                        review_count = len(driver.find_elements(By.CLASS_NAME, 'zPfVt'))
                        try:
                            btn_more = driver.find_element(By.CLASS_NAME, 'place_section.k5tcc')
                            btn_more = btn_more.find_element(By.CLASS_NAME, 'TeItc').click()
                            print('\rReview Crawling Loading: [',
                                  int((review_count / count_all) * 100), '% ] [',
                                  review_count, '/', count_all, ']', end="")
                            time.sleep(1)
                        except NoSuchElementException:
                            print('\rReview Crawling Loading: [ 100 % ]', end="")
                            break
                        except:
                            print('\rReviews More BTN Error')

                        if review_count > 500:
                            break

                    # 리뷰 출력
                    review = ''
                    review_lists = driver.find_elements(By.CLASS_NAME, 'zPfVt')
                    for review_list in review_lists:
                        try:
                            review = review + ' ' + re.compile('[^가-힣]').sub(' ', review_list.text)
                        except:
                            pass
                    print('\nReview Crawling End: ', len(review))
                    reviews.append(review)

                    #CSV 저장
                    df['names'] = names
                    df['reviews'] = reviews
                    try:
                        df.to_csv('../data_naver/naver_data_{}_{}_{}_HHJ.csv'.format(location, i, j),
                                  index=False)
                        print('\n★DataFrame To CSV Success★')
                    except Exception as e:
                        print('Error Code:', e)

                    #드라이버 종료
                    driver.close()
                    driver.quit()

                #변수 초기화
                restaurant_num = 0
            page_num = 1

    except Exception as e:
        print('RETRY CODE: ', e)
        try:
            driver.close()
            driver.quit()
        except:
            continue