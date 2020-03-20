import React from 'react';
import styled from 'styled-components';
import Analyze from '../assets/analyze.svg';

const Container = styled.div`
    width: 800px;
    display: flex;
    align-items: center;
    margin-top: auto;
    padding: 10px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);

    & input {
        width: calc(100% - 40px);
        height: 50px;
        margin-right: 20px;
        font-size: 24px;
        border: none;
    }

    & svg {
        width: 40px;
        height: 40px;
        cursor: pointer;
    }
`;

type Props = {
    setURL: Function;
    onLoad: Function;
};

const Search = (props: Props) => {
    return (
        <Container>
            <input
                type='text'
                placeholder="Analyze a website's privacy policy"
                autoFocus={true}
                onChange={(event: React.ChangeEvent<HTMLInputElement>) => props.setURL(event.target.value)}
                onKeyPress={(event: React.KeyboardEvent<HTMLInputElement>) => {
                    if (event.key === 'Enter') {
                        props.onLoad();
                    }
                }}
            />

            <Analyze onClick={() => props.onLoad()} />
        </Container>
    );
};

export default Search;
